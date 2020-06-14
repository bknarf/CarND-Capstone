#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32
from scipy.spatial import KDTree
from scipy import interpolate

import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.current_velocity = 0
        self.pose = None
        self.stopline_wp_idx = -1

        self.MAX_VELOCITY = (rospy.get_param('waypoint_loader/velocity')* 1000.) / (60. * 60.)
        self.MAX_ACCEL = 10.0


        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoints_2d and self.waypoint_tree:
                #get closest waypoint
                next_waypoint_idx = self.get_next_waypoint_idx()
                self.publish_waypoints(next_waypoint_idx)
            rate.sleep()

    def velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x

    def get_next_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        nearest_idx = self.waypoint_tree.query([x,y],1)[1]
        along_idx = (nearest_idx + 1) % len(self.waypoint_tree.data)
        nearest_xy = np.array(self.waypoint_tree.data[nearest_idx])
        along_xy = np.array(self.waypoint_tree.data[along_idx])
        wp_dir = along_xy-nearest_xy
        ego_xy = np.array([x,y])
        #double signed_dist = b_dir_ego.dot(b_point_nb - b_center_ego);
        signed_dist = np.dot(wp_dir,ego_xy-nearest_xy)
        if signed_dist > 0:
            #ego has already passed nearest_xy
            return along_idx
        else:
            #ego has not passed nearest_xy yet
            return nearest_idx

    def publish_waypoints(self,idx):
        idx = max(0, idx)
        rospy.logwarn("waypoint_updater:  next waypoint index = {0}".format(idx))
        rospy.logwarn("waypoint_updater:  len(base_waypoints.waypoints):{0} wp1:{1} wp2:{2}".format(
            len(self.base_waypoints.waypoints),idx, min(idx + LOOKAHEAD_WPS,len(self.base_waypoints.waypoints))))
        start_dist = self.distance(self.base_waypoints.waypoints, idx, min(idx + LOOKAHEAD_WPS,len(self.base_waypoints.waypoints)))
        x = [ start_dist - 1, start_dist ]
        current_velocity = self.current_velocity
        y = [ current_velocity , current_velocity ]
        if self.stopline_wp_idx > idx and self.stopline_wp_idx < idx + LOOKAHEAD_WPS:
            dist_stop = self.distance(self.base_waypoints.waypoints, self.stopline_wp_idx, idx+LOOKAHEAD_WPS)
            #stop 3m in front of line
            dist_stop -= 3
            x.append(dist_stop)
            y.append(0.0)

        x.append(0)
        y.append(self.base_waypoints.waypoints[idx+LOOKAHEAD_WPS])
        spline_rep = interpolate.splrep(x, y)
        if self.stopline_wp_idx > idx and self.stopline_wp_idx < idx+LOOKAHEAD_WPS :
            #decelerating for traffic light
            for i, wp in enumerate(lane.waypoints):
                p = Waypoint()
                p.pose = lane.waypoints[i].pose
                dist = self.distance(lane.waypoints,i,idx+LOOKAHEAD_WPS)
                vel = interpolate.splev(dist, spline_rep, der=0)
                p.twist.twist.linear.x = vel

        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        if not self.waypoints_2d:
            self.base_waypoints = waypoints
            self.waypoints_2d = []
            for wp in waypoints.waypoints:
                self.waypoints_2d.append([wp.pose.pose.position.x, wp.pose.pose.position.y])
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
