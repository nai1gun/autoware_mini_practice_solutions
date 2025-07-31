#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
import lanelet2
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

        # Initialize goal_point to None
        self.goal_point = None

        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_point_callback, queue_size=10)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def load_lanelet2_map(self, lanelet2_map_path):

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
    
    def goal_point_callback(self, msg):
        # loginfo message about receiving the goal point
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w, msg.header.frame_id)
        
        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        
        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

    def current_pose_callback(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        if not self.goal_point:
            rospy.logwarn("Goal point not received yet.")
            return
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
        # find routing graph
        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)
        if not route:
            rospy.logwarn("No route found from start to goal lanelet.")
            return

        # find shortest path
        path = route.shortestPath()
        # This returns LaneletSequence to a point where a lane change would be necessary to continue
        path_no_lane_change = path.getRemainingLane(start_lanelet)
        if not path_no_lane_change:
            rospy.logwarn("No path found from start to goal lanelet without lane change.")
            return
        # loginfo message about the path
        rospy.loginfo("%s - Found path with %d lanelets from start to goal lanelet without lane change.", rospy.get_name(), len(path_no_lane_change))

    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    print("Starting Lanelet2 Global Planner Node...")
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()