#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import Path, Waypoint
from shapely.geometry import LineString, Point
import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest, distance

class Lanelet2GlobalPlanner:
    """
    A class to implement a global planner using lanelet2 for path planning.
    This class is responsible for planning a path based on lanelet2 maps.
    """
        
    def __init__(self):
        # Load the lanelet2 map
        lanelet2_map_path = rospy.get_param('~lanelet2_map_path')
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise ValueError('Unknown coordinate_transformer for loading the Lanelet2 map ("utm" should be used): ' + coordinate_transformer)

        self.lanelet2_map = load(lanelet2_map_path, projector)

        # Initialize goal_point to None
        self.goal_point = None
        self.graph = None

        # Load output_frame and speed_limit from parameters
        self.speed_limit = rospy.get_param('~speed_limit')  # km/h
        self.output_frame = rospy.get_param('~/planning/lanelet2_global_planner/output_frame')

        # Initialize Distance to goal limit
        self.distance_to_goal_limit = rospy.get_param('~/planning/lanelet2_global_planner/distance_to_goal_limit')  # m

        # traffic rules
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                lanelet2.traffic_rules.Participants.VehicleTaxi)
        # routing graph
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        #Publishers
        self.waypoints_pub = rospy.Publisher('global_path', Path, queue_size=1, latch=True)

        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_point_callback, queue_size=10)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)
    
    def goal_point_callback(self, msg):
        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
        
        if not self.start_lanelet:
            rospy.logwarn("Start lanelet not set. Cannot plan route.")
            return
        route = self.graph.getRoute(self.start_lanelet, goal_lanelet, 0, True)
        if not route:
            rospy.logwarn("No route found from start to goal lanelet.")
            return

        # find shortest path
        path = route.shortestPath()
        # This returns LaneletSequence to a point where a lane change would be necessary to continue
        path_no_lane_change = path.getRemainingLane(self.start_lanelet)
        if not path_no_lane_change:
            rospy.logwarn("No path found from start to goal lanelet without lane change.")
            return
        
        # Convert lanelet sequence to waypoints and publish
        waypoints = self.lanelet_sequence_to_waypoints(path_no_lane_change)
        waypoints = self.align_path_end_with_goal(waypoints, self.goal_point)
        self.publish_global_path(waypoints, msg.header.stamp)

    def current_pose_callback(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)
        # get start and end lanelets
        self.start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]

        if self.goal_point is None:
            return

        distance_to_goal = distance(self.current_location, self.goal_point)
        if distance_to_goal <= self.distance_to_goal_limit:
            self.publish_global_path([], msg.header.stamp)  # Publish an empty path to indicate goal reached

    def lanelet_sequence_to_waypoints(self, lanelet_sequence):
        waypoints = []
        last_point = None
        speed_limit_mps = self.speed_limit * 1000.0 / 3600.0  # Convert km/h to m/s

        for lanelet in lanelet_sequence:
            # Get speed_ref if available, else use speed_limit
            if 'speed_ref' in lanelet.attributes:
                speed = float(lanelet.attributes['speed_ref'])
                speed = min(speed, self.speed_limit)
            else:
                speed = self.speed_limit
            speed_mps = speed * 1000.0 / 3600.0

            # Iterate over centerline points
            for _, point in enumerate(lanelet.centerline):
                # Avoid overlapping points between lanelets
                if last_point is not None and point.x == last_point.x and point.y == last_point.y and point.z == last_point.z:
                    continue
                waypoint = Waypoint()
                waypoint.position.x = point.x
                waypoint.position.y = point.y
                waypoint.position.z = point.z
                waypoint.speed = min(speed_mps, speed_limit_mps)
                waypoints.append(waypoint)
                last_point = point

        return waypoints
    
    def publish_global_path(self, waypoints, timestamp=None):
        if not timestamp:
            timestamp = rospy.Time.now()
        path = Path()
        path.header.frame_id = self.output_frame
        path.header.stamp = timestamp
        path.waypoints = waypoints
        self.waypoints_pub.publish(path)

    def align_path_end_with_goal(self, waypoints, goal_point):
        # Convert waypoints to LineString
        path_line = LineString([(wp.position.x, wp.position.y) for wp in waypoints])
        # Project goal point onto path
        goal_shapely = Point(goal_point.x, goal_point.y)
        projected_dist = path_line.project(goal_shapely)
        projected_point = path_line.interpolate(projected_dist)
        # Overwrite last waypoint with projected point
        waypoints[-1].position.x = projected_point.x
        waypoints[-1].position.y = projected_point.y
        # Update self.goal_point as well
        self.goal_point.x = projected_point.x
        self.goal_point.y = projected_point.y
        return waypoints
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    print("Starting Lanelet2 Global Planner Node...")
    rospy.init_node('lanelet2_global_planner')
    node = Lanelet2GlobalPlanner()
    node.run()