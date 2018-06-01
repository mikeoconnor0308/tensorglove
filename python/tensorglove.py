import argparse
import tensorflow as tf
import glovedata
from glovedata import FEATURES
import tensorglove_osc_server
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=6666, type=int,
                    help='number of training steps')
parser.add_argument('--run_server', default=False, help='whether to run prediction server')
parser.add_argument('--splits', default=10, help='number of splits')


def serving_input_receiver_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat in FEATURES:
        inputs[feat] = tf.placeholder(shape=[None], dtype='float32')

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.items()
    }

    return tf.estimator.export.ServingInputReceiver(features,
                                                    inputs)


def run_server(classifier):
    """
    Runs the OSC server for providing predictions.
    :param classifier: Trained tensorflow classifier
    :return:
    """
    server = tensorglove_osc_server.OscServer("127.0.0.1", 54321, classifier)
    server.run_server()


def main(argv):
    """
    Run the training and serving.
    :param argv: Command line arguments.
    :return:
    """
    global hidden_units
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x_all, train_y_all), (test_x, test_y) = glovedata.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x_all.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))


    kf = KFold(n_splits=args.splits, shuffle=True)
    for train_idx, val_idx in kf.split(train_x_all, train_y_all):
        train_x = pd.DataFrame([train_x_all.iloc[i] for i in train_idx])
        train_y = pd.DataFrame([train_y_all.iloc[i] for i in train_idx])

        # Build 2 hidden layer DNN with 10, 10 units respectively.
        classifier = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=hidden_units,
            # The model must choose between 4 classes.
            n_classes=4,
            model_dir="model_{0}_{1}".format(hidden_units[0], hidden_units[1]))

        # Train the Model.
        classifier.train(
            input_fn=lambda: glovedata.train_input_fn(train_x, train_y,
                                                      args.batch_size),
            steps=args.train_steps)

        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda: glovedata.eval_input_fn(test_x, test_y,
                                                     args.batch_size))

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # perform a sample prediction

    # Generate predictions from the model
    expected = ['Fist']
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
    # the data is expected to be a list of feature values (as it is configured for batching)
    predict_x = dict(zip(FEATURES, [[x] for x in feature_values]))

    predictions = classifier.predict(
        input_fn=lambda: glovedata.eval_input_fn(predict_x,
                                                 labels=None,
                                                 batch_size=args.batch_size))

    template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(glovedata.GESTURES[class_id],
                              100 * probability, expec))

    export_dir = classifier.export_savedmodel(
        export_dir_base="model_export",
        serving_input_receiver_fn=serving_input_receiver_fn)
    print('Exported to:', export_dir)

    if args.run_server:
        run_server(classifier)


if __name__ == '__main__':
    #tf.logging.set_verbosity(tf.logging.INFO)
    #tf.app.run(main)
    hidden_units = [18, 20]
    main(sys.argv)
