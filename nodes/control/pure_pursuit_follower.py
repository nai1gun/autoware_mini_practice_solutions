#!/usr/bin/env python3
import rospy
import numpy as np

from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import VehicleCmd
from shapely.geometry import LineString, Point
from shapely import prepare, distance
from tf.transformations import euler_from_quaternion

class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.path_linstring = None
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 1.0)
        self.wheel_base = rospy.get_param('/vehicle/wheel_base', 3.0)
        print(f"Lookahead distance: {self.lookahead_distance} m, Wheel base: {self.wheel_base} m")

        # Publishers
        self.vehicle_cmd_publisher = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=10)

        # Subscribers
        rospy.Subscriber('path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

        

    def path_callback(self, msg):
        # convert waypoints to shapely linestring
        path_linestring = LineString([(w.position.x, w.position.y) for w in msg.waypoints])
        # prepare path - creates spatial tree, making the spatial queries more efficient
        prepare(path_linestring)
        self.path_linstring = path_linestring

    def current_pose_callback(self, msg):
        current_pose = Point([msg.pose.position.x, msg.pose.position.y])
        if self.path_linstring is None:
            rospy.logwarn("Path not received yet.")
            return
        d_ego_from_path_start = self.path_linstring.project(current_pose)
        
        # using euler_from_quaternion to get the heading angle
        _, _, heading = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        
        # calculate the lookahead point
        lookahead_point = self.path_linstring.interpolate(d_ego_from_path_start + self.lookahead_distance)     
        
        # lookahead point heading calculation
        lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x)

        dynamic_lookahead_distance = distance(current_pose, lookahead_point)

        print(f"Distance from path start: {d_ego_from_path_start:.2f} m")
        print(f"Current heading: {np.degrees(heading):.2f} degrees")
        print(f"Lookahead point: {lookahead_point.x:.2f}, {lookahead_point.y:.2f}")
        print(f"Lookahead heading: {np.degrees(lookahead_heading):.2f} degrees")
        print(f"Dynamic lookahead distance: {dynamic_lookahead_distance:.2f} m")

        steering_angle = np.arctan2(2 * self.wheel_base * np.sin(lookahead_heading - heading), dynamic_lookahead_distance)

        # Publish the vehicle command
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.ctrl_cmd.steering_angle = steering_angle
        vehicle_cmd.ctrl_cmd.linear_velocity = 10.0
        vehicle_cmd.header.stamp = msg.header.stamp
        vehicle_cmd.header.frame_id = "base_link"
        self.vehicle_cmd_publisher.publish(vehicle_cmd)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()