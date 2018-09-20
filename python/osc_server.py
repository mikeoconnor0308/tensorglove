from tensorflow.contrib import predictor
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
import json
import glovedata


class OscServer:
    """
    Simple test class for serving gesture predictions over OSC.
    """

    def predict(self, address, *args):
        """
        OSC message handler for running prediction.
        :param address: OSC message address
        :param args: OSC message arguments, the features to pass to the classifier.
        :return:
        """
        print('Message received', args)
        # construct a tuple of the features, which will be yielded by the generator.
        predict_x = tuple([[float(x)] for x in args])
        # perform the prediction.
        predictions = self.classifier.predict(predict_x)

        for pred_dict in predictions:
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            print('Sending prediction:', hand, class_id)
            # send the prediction back.
            self.client.send_message("/prediction", [hand, int(class_id)])

    def __init__(self, address, port, classifier, left_classifier):
        """
        Initialises the OSC server for providing classifications.
        :param address: IP address to host.
        :param port: Port to use
        :param classifier: Tensorflow classifier.
        :param left_classifier: Classifier for left hand.
        """
        self.address = address
        self.port = port
        self.classifier = classifier
        self.left_classifier = left_classifier
        self.dispatcher = dispatcher.Dispatcher()
        # attach the osc address to the predict method.
        self.dispatcher.map("/predict", self.predict)

        self.server = osc_server.ThreadingOSCUDPServer(
            (address, port), self.dispatcher)
        self.client = udp_client.SimpleUDPClient(address, 54322)
        self.i = 0

    def run_server(self):
        """
        Runs the server.
        :return:
        """
        print("Serving on {}".format(self.server.server_address))
        self.server.serve_forever()