#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32
from lowpass import LowPassFilter
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
        self.published_waypoints = None
        self.published_waypoints_offset = None
        self.last_stopline_wp_idx = None
        self.pose = None
        self.stopline_wp_idx = -1

        self.MAX_VELOCITY = (rospy.get_param('waypoint_loader/velocity')* 1000.) / (60. * 60.)
        self.MAX_ACCEL = 10.0
        tau = 0.5  # 1/2(pi*tau) = cutoff frequency
        ts = 0.02  # sample time

        self.velocity_lpf = LowPassFilter(tau, ts)
        self.current_velocity = 0

        self.last_lane = None


        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoints_2d and self.waypoint_tree:
                #get closest waypoint
                next_waypoint_idx = self.get_next_waypoint_idx()
                self.publish_waypoints(next_waypoint_idx)
            rate.sleep()

    def velocity_cb(self, msg):
        self.current_velocity = self.velocity_lpf.filt(msg.twist.linear.x)

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
        end_idx = min(idx + LOOKAHEAD_WPS,len(self.base_waypoints.waypoints)-1)

        stopping = False
        if self.published_waypoints is not None and self.last_stopline_wp_idx == self.stopline_wp_idx\
                and not (self.stopline_wp_idx > idx and self.stopline_wp_idx < end_idx):
            #reuse and extend the waypoints
            used_up = idx - self.published_waypoints_offset
            self.published_waypoints = self.published_waypoints[used_up - 1:]
            self.published_waypoints_offset = idx
            if self.stopline_wp_idx == -1 and len(self.published_waypoints) < LOOKAHEAD_WPS:
                #cruising and we should copy over some waypoints
                first_copied = idx+len(self.published_waypoints)
                if first_copied < len(self.base_waypoints.waypoints):
                    self.published_waypoints.append(self.base_waypoints.waypoints[first_copied:end_idx])
        else:
            #either stopping, standing still or starting

            start_dist = self.distance(self.base_waypoints.waypoints, idx, end_idx)
            x = [start_dist + 1, start_dist]
            current_velocity = self.current_velocity

            y = [current_velocity, current_velocity]
            if self.stopline_wp_idx > idx and self.stopline_wp_idx < end_idx:
                stopping = True
                dist_stop = self.distance(self.base_waypoints.waypoints, self.stopline_wp_idx, end_idx)
                x.append(max(dist_stop,0.5))
                y.append(0.0)

            if stopping:
                x.append(0)
                y.append(0)

            else:
                x.append(0)
                y.append(min(self.MAX_VELOCITY,self.base_waypoints.waypoints[end_idx].twist.twist.linear.x))

            x.append(-1.0)
            y.append(y[-1])

            x.reverse()
            y.reverse()
            fixed_speed = False
            if (not stopping and abs(y[-1]-y[0]) < 0.5):
                fixed_speed = y[0]
            else:
                spline_rep = interpolate.splrep(x, y, s=0.0)

            rospy.logwarn(
                "waypoint_updater: stopping:{0} fixed_speed:{1} stopline_idx:{2} x:{3} y:{4}".format(stopping, fixed_speed,
                                                                                                     self.stopline_wp_idx,
                                                                                                     x,
                                                                                                     y))
            new_wps = []
            log_vel = []
            for i in range(idx, end_idx):
                p = Waypoint()
                p.pose = self.base_waypoints.waypoints[i].pose
                dist = self.distance(self.base_waypoints.waypoints,i,end_idx)
                if i == idx:
                    vel = current_velocity
                elif fixed_speed:
                    vel = fixed_speed
                else:
                    vel = interpolate.splev(dist, spline_rep, der=0).sum()

                if vel < 0.1 and stopping and dist < 1:
                    vel = 0.0
                else:
                    vel = max(vel , 0.2)
                p.twist.twist.linear.x = vel
                log_vel.append(vel)
                new_wps.append(p)
            self.published_waypoints = new_wps
            self.published_waypoints_offset = idx
            rospy.logwarn(
                "waypoint_updater: velocities:{0} ".format(log_vel))

        lane = Lane()
        lane.waypoints = self.published_waypoints
        self.final_waypoints_pub.publish(lane)
        #exit()

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
        self.stopline_wp_idx = msg.data

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
