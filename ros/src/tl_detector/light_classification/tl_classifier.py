from styx_msgs.msg import TrafficLight
from keras.models import load_model
import os

class TLClassifier(object):
    def __init__(self):
        self.modelpath = os.path.join(os.path.split(os.path.abspath(__file__))[0],"model_96.h5")
        self.model = load_model(self.modelpath)
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        prediction = int(self.model.predict(image, batch_size=1))
        return prediction
