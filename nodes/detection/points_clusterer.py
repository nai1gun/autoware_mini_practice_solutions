#!/usr/bin/env python3
import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from ros_numpy import numpify, msgify

class PointsClusterer:
    def __init__(self):

        # Parameters
        # self.cluster_distance_threshold = rospy.get_param('~cluster_distance_threshold', 0.5)

        # Publishers 

        # Subscribers
        rospy.Subscriber('/detection/lidar/points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

    def points_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        rospy.loginfo("Points shape %d %d", points.shape[0], points.shape[1])


    def run(self):
        # Main loop for clustering points
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer')
    node = PointsClusterer()
    node.run()