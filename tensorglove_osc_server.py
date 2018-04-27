from tensorflow.contrib import predictor
from glovedata import FEATURES,GESTURES

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
        print('Message received', args)
        # expects a list of all the features, in the order of the feature list.
        predict_x = dict(zip(FEATURES, [[float(x)] for x in args]))
        # perform the prediction.
        predictions = self.classifier.predict(
            input_fn=lambda: glovedata.eval_input_fn(predict_x,
                                                     labels=None,
                                                     batch_size=100))

        for pred_dict in predictions:
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            print('Sending prediction:', class_id)
            self.client.send_message("/prediction", int(class_id))

    def __init__(self, address, port, classifier):
        self.address = address
        self.port = port
        self.classifier = classifier

        self.dispatcher = dispatcher.Dispatcher()
        # attach the osc address to the predict method.
        self.dispatcher.map("/predict", self.predict)

        self.server = osc_server.ThreadingOSCUDPServer(
            (address, port), self.dispatcher)
        self.client = udp_client.SimpleUDPClient(address, 54322)

    def run_server(self):
        print("Serving on {}".format(self.server.server_address))
        self.server.serve_forever()