#!/usr/bin/env python3
import rospy

from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import VehicleCmd
from shapely.geometry import LineString, Point
from shapely import prepare, distance

class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.path_linstring = None

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
        print(f"Distance from path start: {d_ego_from_path_start:.2f} m")

        # Test publish
        vehicle_cmd = VehicleCmd()
        vehicle_cmd.ctrl_cmd.steering_angle = 0.2
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