from pythonosc import osc_message_builder
from pythonosc import udp_client
from glovedata import FEATURES
from pythonosc import dispatcher
from pythonosc import osc_server
import json
import time


def prediction(unused_addr, args):
    print("Prediction received: ", args)

feature_values = [0, 0, 0, 1, 0.2298394, 0.9228836, -0.1887825, -0.2445833, 0.03349165, -0.2397906, -0.1800468,
                  0.9533949, -0.07355202, 0.2745002, -0.2481479, 0.9261007, -4.470348E-08, 0.2213377, 1.490116E-08,
                  0.9751973, -2.235174E-08, -5.960464E-08, 4.470348E-08, 1, 0.09148686, -0.09309785, -0.6949129,
                  0.7071485, 7.105427E-15, 7.45058E-08, -0.7191889, 0.6948146, 1.490116E-08, -1.490116E-08,
                  -0.566272, 0.8242185, -2.235174E-08, -5.960464E-08, 4.470348E-08, 1, -3.725291E-08, 5.960464E-08,
                  -0.7009093, 0.7132504, 1.490116E-08, 3.352762E-08, -0.772563, 0.6349381, -7.450572E-09,
                  2.980232E-08, -0.6148145, 0.7886717, -2.235174E-08, -5.960464E-08, 4.470348E-08, 1, -0.06108833,
                  0.06216392, -0.6982422, 0.7105362, -3.352762E-08, 1.490116E-08, -0.7539417, 0.6569413,
                  5.960465E-08, 5.960463E-08, -0.5976215, 0.8017784, -2.235174E-08, -5.960464E-08, 4.470348E-08, 1,
                  -0.1217117, 0.1238547, -0.6902609, 0.7024146, -5.029142E-08, 3.725291E-08, -0.819152, 0.5735765,
                  5.215406E-08, 5.960463E-08, -0.6427875, 0.7660446]
# the data is expected to be a list of feature values (as it is configured for batching
predict_x = dict(zip(FEATURES, [[x] for x in feature_values]))

client = udp_client.SimpleUDPClient("127.0.0.1", 54321)

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/prediction", prediction)
server = osc_server.ThreadingOSCUDPServer(
    ("127.0.0.1", 54322), dispatcher)

while True:
    time.sleep(2)
    json_str = json.dumps(predict_x)
    dict = json.loads(json_str)
    client.send_message("/predict", json_str)
    server.serve_forever()