#!/usr/bin/env python3
import rospy
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import GPSPoint, BasicPoint2d, BoundingBox2d, BasicPoint3d
from lanelet2.geometry import length2d, findNearest, project, findWithin2d

class Lanelet2GlobalPlanner:
    """
    A class to implement a global planner using lanelet2 for path planning.
    This class is responsible for planning a path based on lanelet2 maps.
    """
        
    def __init__(self):
        print("Initializing Lanelet2 Global Planner...")
        # Load the lanelet2 map
        lanelet2_map_path = rospy.get_param('~lanelet2_map_path')
        self.lanelet2_map = self.load_lanelet2_map(lanelet2_map_path)
        print(f"Loaded lanelet2 map from: {lanelet2_map_path}")
        print(f"Type of lanelet2 map: {type(self.lanelet2_map)} ")

    def load_lanelet2_map(self, lanelet2_map_path):
        """
        Load a lanelet2 map from a file and return it
        :param lanelet2_map_path: name of the lanelet2 map file
        :param coordinate_transformer: coordinate transformer
        :param use_custom_origin: use custom origin
        :param utm_origin_lat: utm origin latitude
        :param utm_origin_lon: utm origin longitude
        :return: lanelet2 map
        """

        # get parameters
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise ValueError('Unknown coordinate_transformer for loading the Lanelet2 map ("utm" should be used): ' + coordinate_transformer)

        lanelet2_map = load(lanelet2_map_path, projector)

        return lanelet2_map
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    print("Starting Lanelet2 Global Planner Node...")
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()