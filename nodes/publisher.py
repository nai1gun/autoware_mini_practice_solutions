#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

rospy.init_node('publisher')
rateValue = rospy.get_param('~rate', 2)  # Default rate is 2 Hz
rate = rospy.Rate(rateValue)
pub = rospy.Publisher('/message', String, queue_size=10)

while not rospy.is_shutdown():
    message = rospy.get_param('~message', 'Hello World!')
    pub.publish(message)
    rate.sleep()