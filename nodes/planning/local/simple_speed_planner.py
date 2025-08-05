#!/usr/bin/env python3

import rospy
import math
import message_filters
import traceback
import shapely
import numpy as np
import threading
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify
from autoware_mini.msg import Path, Log
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from autoware_mini.geometry import project_vector_to_heading, get_distance_between_two_points_2d
from shapely.geometry import LineString, Point, Polygon


class SpeedPlanner:

    def __init__(self):

        # parameters
        self.default_deceleration = rospy.get_param("default_deceleration")
        rospy.loginfo("Default deceleration: %f", self.default_deceleration)
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
                # rospy.logwarn_throttle(10, "%s - No collision points or current position/speed available", rospy.get_name())
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

                    # Calculate target velocity using v = sqrt(v0^2 + 2*a*s)
                    v0 = math.hypot(pt['vx'], pt['vy'])
                    a = abs(self.default_deceleration)
                    s = dist_along_path - self.distance_to_car_front - pt['distance_to_stop']
                    under_sqrt = v0**2 + 2*a*s
                    v = math.sqrt(max(0.0, under_sqrt))
                    target_velocities.append(v)

                    # Get heading at this distance
                    heading = self.get_heading_at_distance(local_path_linestring, dist_along_path)
                    # Project velocity vector to heading
                    velocity_vector = Vector3(pt['vx'], pt['vy'], pt['vz'])
                    projected_velocity = self.project_vector_to_heading(heading, velocity_vector)
                    collision_point_velocities.append(projected_velocity)
                    # Print actual speed and projected speed
                    print(f"Collision point at distance {dist_along_path:.2f}: actual speed = {v0:.2f}, speed along heading = {projected_velocity:.2f}")

                # rospy.loginfo("Target velocities # along path: %s", target_velocities)
                # Find the minimum target velocity (most restrictive)
                if target_velocities:
                    # Find the index of the minimum target velocity
                    min_idx = int(np.argmin(target_velocities))
                    min_target_velocity = target_velocities[min_idx]
                    # Distance from local path start to the stopping point (where the car should stop)
                    stopping_point_distance = collision_points_distances[min_idx] - collision_points[min_idx]['distance_to_stop']
                    stopping_point_distance = max(0.0, stopping_point_distance)
                    # Distance from car front to the obstacle when stopped (should be close to braking_safety_distance_obstacle)
                    closest_object_distance = collision_points_distances[min_idx]
                    collision_point_category = int(collision_points[min_idx]['category'])
                    closest_object_velocity = math.hypot(collision_points[min_idx]['vx'], collision_points[min_idx]['vy'])
                else:
                    min_target_velocity = 0.0
                    stopping_point_distance = 0.0
                    closest_object_distance = 0.0
                    collision_point_category = 4
                    closest_object_velocity = 0.0

                # rospy.loginfo("Minimum target velocity: %f, closest object distance: %f, category: %d, velocity: %f",   
                #              min_target_velocity, closest_object_distance, collision_point_category, closest_object_velocity)

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

                # print("Collision points distances along path:", collision_points_distances)
            
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