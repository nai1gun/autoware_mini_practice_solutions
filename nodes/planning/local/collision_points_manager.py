#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray, TrafficLightResultArray
from sensor_msgs.msg import PointCloud2
from shapely.geometry import LineString, Polygon
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector

DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),
    ('deceleration_limit', np.float32),
    ('category', np.int32)
])

class CollisionPointsManager:

    def __init__(self):

        # Parameters related to lanelet2 map loading
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        # parameters
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_stopline = rospy.get_param("~braking_safety_distance_stopline")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        self.goal_point = None

        # variables
        self.detected_objects = None
        self.stopline_statuses = {}

        # Lock for thread safety
        self.lock = threading.Lock()

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        lanelet2_map = load(lanelet2_map_path, projector)

        # Extract all stop lines and signals from the lanelet2 map
        self.stoplines = get_stoplines(lanelet2_map)

        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=1, tcp_nodelay=True)

    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects
        collision_points = np.array([], dtype=DTYPE)

        # Check for empty path or no detected objects
        if not msg.waypoints or len(msg.waypoints) == 0:
            empty_msg = msgify(PointCloud2, collision_points)
            empty_msg.header = msg.header
            self.local_path_collision_pub.publish(empty_msg)
            return
        
        local_path_linestring = LineString([(waypoint.position.x, waypoint.position.y) for waypoint in msg.waypoints])

        # Buffer the local path
        path_buffer = local_path_linestring.buffer(self.safety_box_width / 2, cap_style='flat')  # flat caps
        shapely.prepare(path_buffer)

        if detected_objects:
            for obj in detected_objects:
                if not hasattr(obj, 'convex_hull') or len(obj.convex_hull) < 9:
                    continue  # Not enough points for a polygon (3 points * 3 coords)

                # obj.convex_hull is a flat list: [x1, y1, z1, x2, y2, z2, ...]
                hull_xy = [(obj.convex_hull[i], obj.convex_hull[i+1]) for i in range(0, len(obj.convex_hull), 3)]
        
                try:
                    obj_polygon = Polygon(hull_xy)
                except Exception:
                    continue  # Skip invalid polygons

                # Check intersection
                if not path_buffer.intersects(obj_polygon):
                    continue

                intersection = path_buffer.intersection(obj_polygon)
                intersection_points = shapely.get_coordinates(intersection)

                # Calculate object speed
                object_speed = math.hypot(obj.velocity.x, obj.velocity.y)

                for x, y in intersection_points:
                    collision_points = np.append(
                        collision_points,
                        np.array([(
                            x, y, obj.centroid.z,
                            obj.velocity.x, obj.velocity.y, obj.velocity.z,
                            self.braking_safety_distance_obstacle,
                            np.inf,
                            3 if object_speed < self.stopped_speed_limit else 4
                        )], dtype=DTYPE)
                    )

        # --- Traffic light stopline collision points ---
        for stopline_id, stopline_geom in self.stoplines.items():
            # Only consider stoplines with status RED or YELLOW
            status = self.stopline_statuses.get(stopline_id, None)
            if status != 0:  # Only STOP (RED or YELLOW)
                continue

            # Check intersection with buffered path
            if not path_buffer.intersects(stopline_geom):
                continue

            intersection = path_buffer.intersection(stopline_geom)
            intersection_points = shapely.get_coordinates(intersection)

            for x, y in intersection_points:
                collision_points = np.append(
                    collision_points,
                    np.array([(
                        x, y, 0.0,  # z=0 for stopline
                        0.0, 0.0, 0.0,  # velocity
                        self.braking_safety_distance_stopline,
                        np.inf,
                        2  # category 2: traffic light stop line
                    )], dtype=DTYPE)
                )

        # --- Goal point collision point ---
        if self.goal_point is not None:
            # Create a small buffer around the goal point for intersection check
            goal_shapely = shapely.geometry.Point(self.goal_point[0], self.goal_point[1])
            goal_buffer = goal_shapely.buffer(self.safety_box_width / 2)
            if path_buffer.intersects(goal_buffer):
                # Use goal point as collision point
                collision_points = np.append(
                    collision_points,
                    np.array([(
                        self.goal_point[0], self.goal_point[1], self.goal_point[2],
                        0.0, 0.0, 0.0,  # velocity
                        self.braking_safety_distance_goal,
                        np.inf,
                        1  # category 1: goal point
                    )], dtype=DTYPE)
                )
            else:
                rospy.logwarn("No goal point intersect %s with %s", path_buffer, goal_buffer)


        # Publish collision points
        collision_msg = msgify(PointCloud2, collision_points)
        collision_msg.header = msg.header
        self.local_path_collision_pub.publish(collision_msg)

    def traffic_light_status_callback(self, msg):
        # Store the latest status for each stopline
        self.stopline_statuses = {res.stopline_id: res.recognition_result for res in msg.results}

    def global_path_callback(self, msg):
        # Save the last waypoint as the goal point
        if msg.waypoints:
            goal_wp = msg.waypoints[-1]
            self.goal_point = (goal_wp.position.x, goal_wp.position.y, goal_wp.position.z)
        else:
            self.goal_point = None

    def run(self):
        rospy.spin()

def get_stoplines(lanelet2_map):
    """
    Add all stop lines to a dictionary with stop_line id as key and stop_line as value
    :param lanelet2_map: lanelet2 map
    :return: {stop_line_id: stopline, ...}
    """

    stoplines = {}
    for line in lanelet2_map.lineStringLayer:
        if line.attributes:
            if line.attributes["type"] == "stop_line":
                # add stoline to dictionary and convert it to shapely LineString
                stoplines[line.id] = LineString([(p.x, p.y) for p in line])

    return stoplines

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()