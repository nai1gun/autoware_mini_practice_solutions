#!/usr/bin/env python3
import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from ros_numpy import numpify, msgify
from sklearn.cluster import DBSCAN

class PointsClusterer:
    def __init__(self):

        # Parameters
        self.cluster_epsilon = rospy.get_param('~/detection/lidar/points_clusterer/cluster_epsilon')
        self.cluster_min_size = rospy.get_param('~/detection/lidar/points_clusterer/cluster_min_size')

        # Initialize the DBSCAN clusterer
        self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size)
        # Publishers 

        # Subscribers
        rospy.Subscriber('/detection/lidar/points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

    def points_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        rospy.loginfo("Points shape %d %d", points.shape[0], points.shape[1])

        # Perform clustering
        labels = self.clusterer.fit_predict(points)
        rospy.loginfo("Labels shape: %s", labels.shape)

        # Assert that the number of labels matches the number of points
        if points.shape[0] != labels.shape[0]:
            raise AssertionError(f"Number of points ({points.shape[0]}) does not match number of labels ({labels.shape[0]})")

    def run(self):
        # Main loop for clustering points
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer')
    node = PointsClusterer()
    node.run()