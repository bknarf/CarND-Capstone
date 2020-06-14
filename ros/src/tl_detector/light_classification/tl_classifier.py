from styx_msgs.msg import TrafficLight
from keras.models import load_model
import numpy as np
import os
import tensorflow as tf
import rospy

class TLClassifier(object):
    def __init__(self):
        self.modelpath = os.path.join(os.path.split(os.path.abspath(__file__))[0],"model_96.h5")
        self.model = load_model(self.modelpath)
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            image_array = np.asarray(image)
            image_array = image_array[None, :, :, :]
            prediction = self.model.predict(image_array, batch_size=1)
            rospy.logwarn(
                "tl_classifier:  prediction:{0}".format(
                    prediction))
            prediction = np.where(np.isclose(prediction, 1.0))[0][0]
            rospy.logwarn(
                "tl_classifier:  prediction:{0}".format(
                    prediction))
            return prediction
