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
        self.cluster_pub = rospy.Publisher('/detection/lidar/points_clustered', PointCloud2, queue_size=1, tcp_nodelay=True)

        # Subscribers
        rospy.Subscriber('/detection/lidar/points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

    def points_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)

        # Perform clustering
        labels = self.clusterer.fit_predict(points)

        # Assert that the number of labels matches the number of points
        if points.shape[0] != labels.shape[0]:
            raise AssertionError(f"Number of points ({points.shape[0]}) does not match number of labels ({labels.shape[0]})")

        # Concatenate points and labels
        points_with_labels = np.hstack((points, labels.reshape(-1, 1)))

        # Filter out noise points (label == -1)
        clustered_points = points_with_labels[points_with_labels[:, -1] != -1]

        # convert labelled points to PointCloud2 format
        data = unstructured_to_structured(clustered_points, dtype=np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('label', np.int32)
        ]))

        # publish clustered points message
        cluster_msg = msgify(PointCloud2, data)
        cluster_msg.header.stamp = msg.header.stamp
        cluster_msg.header.frame_id = msg.header.frame_id
        self.cluster_pub.publish(cluster_msg)

    def run(self):
        # Main loop for clustering points
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer')
    node = PointsClusterer()
    node.run()