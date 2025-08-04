#!/usr/bin/env python3

import rospy
import numpy as np

from shapely import MultiPoint
from tf2_ros import TransformListener, Buffer, TransformException
from numpy.lib.recfunctions import structured_to_unstructured
from ros_numpy import numpify

from sensor_msgs.msg import PointCloud2
from autoware_mini.msg import DetectedObjectArray, DetectedObject
from std_msgs.msg import ColorRGBA

BLUE80P = ColorRGBA(0.0, 0.0, 1.0, 0.8)

class ClusterDetector:
    def __init__(self):
        self.min_cluster_size = rospy.get_param('~min_cluster_size')
        self.output_frame = rospy.get_param('/detection/output_frame')
        self.transform_timeout = rospy.get_param('~transform_timeout')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.objects_pub = rospy.Publisher('detected_objects', DetectedObjectArray, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('points_clustered', PointCloud2, self.cluster_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)

        rospy.loginfo("%s - initialized", rospy.get_name())


    def cluster_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        labels = data['label']
        unique_labels = np.unique(labels)

        if msg.header.frame_id != self.output_frame:
            try:
                transform = self.tf_buffer.lookup_transform(self.output_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(self.transform_timeout))
            except (TransformException, rospy.ROSTimeMovedBackwardsException) as e:
                rospy.logwarn("%s - %s", rospy.get_name(), e)
                return
            tf_matrix = numpify(transform.transform).astype(np.float32)
            # make copy of points
            points = points.copy()
            # add homogeneous coordinate
            points = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # transform points to target frame
            points = points.dot(tf_matrix.T)

        detected_array = DetectedObjectArray()
        detected_array.header.stamp = msg.header.stamp
        detected_array.header.frame_id = self.output_frame
        
        # Check if there are any points (clusters/objects)
        if points.shape[0] == 0:
            # No objects, publish empty array
            self.objects_pub.publish(detected_array)
            return

        for label in unique_labels:

            if label == -1:  # Skip noise points
                continue

            mask = (labels == label)
            # select points for one object from an array using a mask
            # rows are selected using a binary mask, and only the first 3 columns are selected: x, y, and z coordinates
            points3d = points[mask,:3]
            cluster_size = points3d.shape[0]
            if(cluster_size < self.min_cluster_size):
                continue

            # Calculate centroid
            centroid = np.mean(points3d, axis=0)

            # Create DetectedObject
            obj = DetectedObject()
            obj.id = label
            obj.label = "unknown"
            obj.color = BLUE80P
            obj.valid = True
            obj.position_reliable = True
            obj.velocity_reliable = False
            obj.acceleration_reliable = False

            obj.centroid.x = float(centroid[0])
            obj.centroid.y = float(centroid[1])
            obj.centroid.z = float(centroid[2])

            # Convex hull (2D)
            points_2d = MultiPoint(points3d[:, :2])
            hull = points_2d.convex_hull
            convex_hull_points = [a for hull in [[x, y, centroid[2]] for x, y in hull.exterior.coords] for a in hull]
            obj.convex_hull = convex_hull_points

            detected_array.objects.append(obj)

        self.objects_pub.publish(detected_array)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('cluster_detector', log_level=rospy.INFO)
    node = ClusterDetector()
    node.run()