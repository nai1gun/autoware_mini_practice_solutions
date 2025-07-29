#!/usr/bin/env python3

import math
import rospy

from tf.transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
from pyproj import CRS, Transformer, Proj

from novatel_oem7_msgs.msg import INSPVA
from geometry_msgs.msg import PoseStamped, TwistStamped, Quaternion, TransformStamped

class Localizer:
    def __init__(self):

        # Parameters
        self.undulation = rospy.get_param('undulation')
        utm_origin_lat = rospy.get_param('utm_origin_lat')
        utm_origin_lon = rospy.get_param('utm_origin_lon')

        # Internal variables
        self.crs_wgs84 = CRS.from_epsg(4326)
        self.crs_utm = CRS.from_epsg(25835)
        self.utm_projection = Proj(self.crs_utm)

        # Subscribers
        rospy.Subscriber('/novatel/oem7/inspva', INSPVA, self.transform_coordinates)

        # Publishers
        self.current_pose_pub = rospy.Publisher('current_pose', PoseStamped, queue_size=10)
        self.current_velocity_pub = rospy.Publisher('current_velocity', TwistStamped, queue_size=10)
        self.br = TransformBroadcaster()

        # create a coordinate transformer
        self.transformer = Transformer.from_crs(self.crs_wgs84, self.crs_utm)
        self.origin_x, self.origin_y = self.transformer.transform(utm_origin_lat, utm_origin_lon)

    # convert azimuth to yaw angle
    def convert_azimuth_to_yaw(self, azimuth):
        """
        Converts azimuth to yaw. Azimuth is CW angle from the north. Yaw is CCW angle from the East.
        :param azimuth: azimuth in radians
        :return: yaw in radians
        """
        yaw = -azimuth + math.pi/2
        # Clamp within 0 to 2 pi
        if yaw > 2 * math.pi:
            yaw = yaw - 2 * math.pi
        elif yaw < 0:
            yaw += 2 * math.pi

        return yaw

    def transform_coordinates(self, msg):
        # print(msg.latitude, msg.longitude)
        transformed_x, transformed_y = self.transformer.transform(msg.latitude, msg.longitude)
        substracted_x = transformed_x - self.origin_x
        substracted_y = transformed_y - self.origin_y
        # print(f"x:{substracted_x} :{substracted_y}")
        # publish current pose
        current_pose_msg = PoseStamped()
        current_pose_msg.header.stamp = msg.header.stamp
        current_pose_msg.header.frame_id = "map"
        current_pose_msg.pose.position.x = substracted_x
        current_pose_msg.pose.position.y = substracted_y
        current_pose_msg.pose.position.z = msg.height - self.undulation

        # calculate azimuth correction
        azimuth_correction = self.utm_projection.get_factors(msg.longitude, msg.latitude).meridian_convergence
        
        azimuth_rad = math.radians(msg.azimuth)
        # print(f"Azimuth:{msg.azimuth}")
        # print(f"Azimuth rad:{azimuth_rad}")
        # print(f"Correction:{azimuth_correction}")
        
        yaw = self.convert_azimuth_to_yaw(azimuth_rad - azimuth_correction)

        # Convert yaw to quaternion
        x, y, z, w = quaternion_from_euler(0, 0, yaw)
        orientation = Quaternion(x, y, z, w)

        current_pose_msg.pose.orientation = orientation
        self.current_pose_pub.publish(current_pose_msg)

        current_velocity = TwistStamped()
        current_velocity.header.stamp = msg.header.stamp
        current_velocity.header.frame_id = "base_link"
        current_velocity.twist.linear.x = msg.north_velocity
        current_velocity.twist.linear.y = msg.east_velocity

        self.current_velocity_pub.publish(current_velocity)

        # create a transform message
        t = TransformStamped()

        # fill in the transform message - t
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"

        t.transform.translation.x = substracted_x
        t.transform.translation.y = substracted_y
        t.transform.translation.z = msg.height - self.undulation

        t.transform.rotation = orientation

        # publish transform
        self.br.sendTransform(t)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('localizer')
    node = Localizer()
    node.run()