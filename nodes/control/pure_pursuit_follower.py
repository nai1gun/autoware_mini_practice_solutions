#!/usr/bin/env python3
import rospy

from autoware_mini.msg import Path
from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import VehicleCmd

class PurePursuitFollower:
    def __init__(self):

        # Parameters

        # Publishers
        self.vehicle_cmd_publisher = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=10)

        # Subscribers
        rospy.Subscriber('path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

        

    def path_callback(self, msg):
        # TODO
        pass

    def current_pose_callback(self, msg):
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