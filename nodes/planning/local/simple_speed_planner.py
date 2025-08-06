#!/usr/bin/env python3

import rospy
import math
import message_filters
import traceback
import shapely
import numpy as np
import threading
from ros_numpy import numpify
from autoware_mini.msg import Path
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from shapely.geometry import LineString, Point


class SpeedPlanner:

    def __init__(self):

        # parameters
        self.default_deceleration = rospy.get_param("default_deceleration")
        self.braking_reaction_time = rospy.get_param("braking_reaction_time")
        synchronization_queue_size = rospy.get_param("~synchronization_queue_size")
        synchronization_slop = rospy.get_param("~synchronization_slop")
        self.distance_to_car_front = rospy.get_param("distance_to_car_front")

        # variables
        self.collision_points = None
        self.current_position = None
        self.current_speed = None

        # Lock for thread safety
        self.lock = threading.Lock()

        # publishers
        self.local_path_pub = rospy.Publisher('local_path', Path, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/localization/current_velocity', TwistStamped, self.current_velocity_callback, queue_size=1, tcp_nodelay=True)

        collision_points_sub = message_filters.Subscriber('collision_points', PointCloud2, tcp_nodelay=True)
        local_path_sub = message_filters.Subscriber('extracted_local_path', Path, tcp_nodelay=True)

        ts = message_filters.ApproximateTimeSynchronizer([collision_points_sub, local_path_sub], queue_size=synchronization_queue_size, slop=synchronization_slop)

        ts.registerCallback(self.collision_points_and_path_callback)

    def current_velocity_callback(self, msg):
        self.current_speed = msg.twist.linear.x

    def current_pose_callback(self, msg):
        self.current_position = shapely.Point(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)

    def collision_points_and_path_callback(self, collision_points_msg, local_path_msg):
        try:
            with self.lock:
                collision_points = numpify(collision_points_msg) if len(collision_points_msg.data) > 0 else np.array([])
                current_position = self.current_position
                current_speed = self.current_speed

            if current_position is None or current_speed is None:
                return
            
            path = Path()
            path.header = local_path_msg.header
            path.waypoints = []
            path.closest_object_distance = 0.0 # Distance to the collision point with lowest target velocity (also closest object for now)
            path.closest_object_velocity = 0 # Velocity of the collision point with lowest target velocity (0)
            path.is_blocked = True
            path.stopping_point_distance = 0.0 # Stopping point distance can be set to the distance to the closest object for now
            path.collision_point_category = 4 # Category of collision point with lowest target velocity
            
            if len(collision_points) == 0:
                rospy.logwarn_throttle(10, "%s - No collision points available", rospy.get_name())
                path.waypoints = local_path_msg.waypoints
            else:
                # 1. Create a LineString from the local path waypoints
                local_path_xy = [(wp.position.x, wp.position.y) for wp in local_path_msg.waypoints]
                local_path_linestring = LineString(local_path_xy)

                # 2. Calculate distances along the path for each collision point
                collision_points_distances = []
                target_velocities = []
                collision_point_velocities = []

                for pt in collision_points:
                    pt_xy = Point(pt['x'], pt['y'])
                    dist_along_path = local_path_linestring.project(pt_xy)
                    collision_points_distances.append(dist_along_path)

                    # Get heading at this distance
                    heading = self.get_heading_at_distance(local_path_linestring, dist_along_path)
                    # Project velocity vector to heading
                    velocity_vector = Vector3(pt['vx'], pt['vy'], pt['vz'])
                    projected_velocity = self.project_vector_to_heading(heading, velocity_vector)
                    collision_point_velocities.append(projected_velocity)

                # Convert to numpy arrays for vectorized math
                collision_points_distances = np.array(collision_points_distances)
                collision_point_velocities = np.array(collision_point_velocities)
                a = abs(self.default_deceleration)
                s = collision_points_distances - self.distance_to_car_front - np.array([pt['distance_to_stop'] for pt in collision_points])
                under_sqrt = collision_point_velocities**2 + 2 * a * s
                target_velocities = np.sqrt(np.maximum(0.0, under_sqrt))

                # Find the minimum target velocity (most restrictive)
                if len(target_velocities) > 0:
                    min_idx = int(np.argmin(target_velocities))
                    min_target_velocity = target_velocities[min_idx]
                    closest_object_distance = collision_points_distances[min_idx]
                    closest_object_velocity = collision_point_velocities[min_idx]
                    stopping_point_distance = collision_points_distances[min_idx] - collision_points[min_idx]['distance_to_stop']
                    stopping_point_distance = max(0.0, stopping_point_distance)
                    collision_point_category = int(collision_points[min_idx]['category'])
                else:
                    min_target_velocity = 0.0
                    stopping_point_distance = 0.0
                    closest_object_distance = 0.0
                    collision_point_category = 4
                    closest_object_velocity = 0.0

                # Overwrite local path waypoint velocities with the minimum target velocity
                path.waypoints = local_path_msg.waypoints
                for wp in path.waypoints:
                    wp.speed = min(min_target_velocity, wp.speed)

                # Assign the additional attributes
                path.closest_object_distance = closest_object_distance
                path.closest_object_velocity = closest_object_velocity
                path.is_blocked = True
                path.stopping_point_distance = stopping_point_distance
                path.collision_point_category = collision_point_category
            
            self.local_path_pub.publish(path)

        except Exception as e:
            rospy.logerr_throttle(10, "%s - Exception in callback: %s", rospy.get_name(), traceback.format_exc())


    def get_heading_at_distance(self, linestring, distance):
        """
        Get heading of the path at a given distance
        :param distance: distance along the path
        :param linestring: shapely linestring
        :return: heading angle in radians
        """

        point_after_object = linestring.interpolate(distance + 0.1)
        # if distance is negative it is measured from the end of the linestring in reverse direction
        point_before_object = linestring.interpolate(max(0, distance - 0.1))

        # get heading between two points
        return math.atan2(point_after_object.y - point_before_object.y, point_after_object.x - point_before_object.x)


    def project_vector_to_heading(self, heading_angle, vector):
        """
        Project vector to heading
        :param heading_angle: heading angle in radians
        :param vector: vector
        :return: projected vector
        """

        return vector.x * math.cos(heading_angle) + vector.y * math.sin(heading_angle)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('speed_planner')
    node = SpeedPlanner()
    node.run()