#!/usr/bin/env python
import datetime

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray
from styx_msgs.msg import TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np
from scipy.spatial import KDTree
import os

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_lights = []
        for counter, stl in enumerate(self.config["stop_line_positions"]):
            self.stop_lights.append(StopLight("tl{0}".format(counter), np.array(stl)))
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN

        self.waypoint_tree = None
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        waypoints_2d = []
        for wp in waypoints.waypoints:
            waypoints_2d.append(np.array([wp.pose.pose.position.x, wp.pose.pose.position.y]))
        waypoint_tree = KDTree(waypoints_2d)
        for tl in self.stop_lights:
            tl.set_waypoint_tree(waypoint_tree)
        self.waypoint_tree = waypoint_tree

    def traffic_cb(self, msg):
        for tlm, tl in zip(msg.lights, self.stop_lights):
            tl.set_light_position(np.array([tlm.pose.pose.position.x,tlm.pose.pose.position.y]))
            tl.set_simstate(tlm.state)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.camera_image = cv_image
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """


        ego_wp_idx = self.get_next_waypoint_idx()
        relevant_tls = []
        for tl in self.stop_lights:
            if tl.is_relevant(ego_wp_idx):
                relevant_tls.append(tl)

        if len(relevant_tls) == 0:
            #no relevant traffic light
            return None, TrafficLight.UNKNOWN
        elif len(relevant_tls) > 1:
            rospy.logwarn(
                "tl_detector:  found more than one relevant traffic light. len(relevant_tls) == {0}".format(
                    len(relevant_tls)))

        #the traffic light decides if an image should be captured or not
        relevant_tls[0].capture_img(self.camera_image)
        state = self.light_classifier.get_classification(self.camera_image)
        return relevant_tls[0].line_waypoint_idx, state

    def get_next_waypoint_idx(self):
        if (self.pose):
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
        else:
            return 0

class StopLight:
    def __init__(self, name, line_position):
        self.name = name
        #as 2D numpy array
        self.line_position = line_position
        self.light_position = None
        self.approach_direction = None
        self.waypoint_tree = None
        self.line_waypoint_idx = None
        self.visible_not_relevant_wpidxs = None
        self.visible_relevant_wpidxs = None

        self.view_distance = 200
        #45 deg left and right of -approach direction
        self.half_viewing_angle = np.pi / 4.0
        self.simstate = None
        self.capture_images = True
        self.capture_every_X_image = 3
        self.capture_counter = 0
        self.capture_image_path = "~/captured_images"

    def set_light_position(self, light_position):
        self.light_position = light_position
        self.approach_direction = self.light_position - self.line_position
        self.approach_direction /= self.approach_direction.sum()
        self.find_waypoint_idxs()

    def set_waypoint_tree(self, waypoint_tree):
        self.waypoint_tree = waypoint_tree
        self.find_waypoint_idxs()

    def set_simstate(self, state):
        self.simstate = state

    def find_waypoint_idxs(self):
        if not (self.line_position is None or self.waypoint_tree is None or self.light_position is None):
            if not self.line_waypoint_idx:
                nearest_line_idx = self.waypoint_tree.query(self.line_position,1)[1]
                nearest_line_xy = np.array(self.waypoint_tree.data[nearest_line_idx])
                # double signed_dist = b_dir_ego.dot(b_point_nb - b_center_ego);
                signed_dist = np.dot(self.approach_direction, self.line_position - nearest_line_xy)
                if signed_dist < 0:
                    # the distance from waypoint in approach direction to line position is negative
                    # the nearest waypoint is behind the stop line
                    nearest_line_idx -= 1
                self.line_waypoint_idx = nearest_line_idx

            if None in (self.visible_relevant_wpidxs,self.visible_not_relevant_wpidxs):
                nearest_light_idx = self.waypoint_tree.query(self.light_position,1)[1]

                #these waypoints are between stopline and traffic light
                self.visible_not_relevant_wpidxs = []
                for i in range(self.line_waypoint_idx,nearest_light_idx,1):
                    dir_at_wp = np.array(self.waypoint_tree.data[i])-np.array(self.waypoint_tree.data[i-1])
                    dir_at_wp /= np.sum(dir_at_wp)
                    angle_between = np.arccos(np.dot(dir_at_wp,self.approach_direction))
                    if angle_between < np.pi/2 and angle_between > -np.pi/2 :
                        self.visible_not_relevant_wpidxs.append(i)
                    else:
                        rospy.logwarn("tl_detector:  found a traffic light, which is close to a waypoint not leading to the stopsign. line_position={0}".format(self.line_position))
                        self.visible_not_relevant_wpidxs.clear()
                        break
                angle_between = 0
                distance = 0
                self.visible_relevant_wpidxs = []
                idx = self.line_waypoint_idx+1
                while distance < self.view_distance and angle_between < self.half_viewing_angle and angle_between > -self.half_viewing_angle:
                    self.visible_relevant_wpidxs.append(idx)
                    current_xy = np.array(self.waypoint_tree.data[idx])
                    along_xy = np.array(self.waypoint_tree.data[idx+1])
                    travel_dir = along_xy - current_xy
                    travel_dir /= np.sum(travel_dir)
                    distance = np.linalg.norm(current_xy - self.light_position)
                    angle_between = np.arccos(np.dot(travel_dir, self.approach_direction))

    def is_relevant(self,wp_idx):
        return wp_idx in self.visible_relevant_wpidxs

    def is_visible(self,wp_idx):
        return wp_idx in self.visible_relevant_wpidxs or self.visible_not_relevant_wpidxs

    def capture_img(self, img):

        if self.capture_images :
            if not os.path.exists(self.capture_image_path):
                os.makedirs(self.capture_image_path)
            if self.capture_counter % self.capture_every_X_image == 0:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = "#".join([self.name , str(self.simstate) , ts])+".jpg"
                path = os.path.join(self.capture_image_path,filename)
                cv2.imwrite(filename,img)
            self.capture_counter += 1

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
