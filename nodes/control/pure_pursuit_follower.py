#!/usr/bin/env python3
import rospy
import numpy as np

from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import VehicleCmd
from shapely.geometry import LineString, Point
from shapely import prepare, distance
from tf.transformations import euler_from_quaternion
from scipy.interpolate import interp1d

class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.path_linstring = None
        self.lookahead_distance = rospy.get_param('~lookahead_distance', 1.0)
        self.wheel_base = rospy.get_param('/vehicle/wheel_base', 3.0)
        self.distance_to_velocity_interpolator = None
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

        # Create a distance-to-velocity interpolator for the path
        # collect waypoint x and y coordinates
        waypoints_xy = np.array([(w.position.x, w.position.y) for w in msg.waypoints])
        # Calculate distances between points
        distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0)**2, axis=1)))
        # add 0 distance in the beginning
        distances = np.insert(distances, 0, 0)
        # Extract velocity values at waypoints
        velocities = np.array([w.speed for w in msg.waypoints])
        # Create an interpolator for distance to velocity
        self.distance_to_velocity_interpolator = interp1d(distances, velocities, kind='linear', bounds_error=False, fill_value=0.0)

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

        # Calculate the velocity based on the distance from the path start
        linear_velocity = 0.0
        if self.distance_to_velocity_interpolator is not None:
            linear_velocity = self.distance_to_velocity_interpolator(d_ego_from_path_start)

        print(f"Linear velocity: {linear_velocity:.2f} m")
            

        # Publish the vehicle command
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.ctrl_cmd.steering_angle = steering_angle
        vehicle_cmd.ctrl_cmd.linear_velocity = linear_velocity
        vehicle_cmd.header.stamp = msg.header.stamp
        vehicle_cmd.header.frame_id = "base_link"
        self.vehicle_cmd_publisher.publish(vehicle_cmd)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()