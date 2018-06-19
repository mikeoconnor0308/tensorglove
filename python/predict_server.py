"""
Script for running the trained DNNClassifier for predictions.
"""

import argparse
import tensorflow as tf
import glovedata
from glovedata import FEATURES_LEFT, FEATURES_RIGHT
import osc_server
import fastpredict
import training

parser = argparse.ArgumentParser()
parser.add_argument('--run_server', default=True, help='whether to run prediction server')
parser.add_argument('--model_dir_left', default="model_left", help='directory model for left hand was saved in.')
parser.add_argument('--model_dir_right', default="model_right", help='directory model for right hand was saved in.')

def generator_evaluation_fn_left(generator):
    """ Input function for numeric feature columns using the generator. """

    def _inner_input_fn():
        datatypes = tuple(len(FEATURES_LEFT) * [tf.float32])
        dataset = tf.data.Dataset().from_generator(generator, output_types=datatypes).batch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        feature_dict = dict(zip(FEATURES_LEFT, features))
        return feature_dict

    return _inner_input_fn


def generator_evaluation_fn_right(generator):
    """ Input function for numeric feature columns using the generator. """

    def _inner_input_fn():
        datatypes = tuple(len(FEATURES_RIGHT) * [tf.float32])
        dataset = tf.data.Dataset().from_generator(generator, output_types=datatypes).batch(1)
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()
        feature_dict = dict(zip(FEATURES_RIGHT, features))
        return feature_dict

    return _inner_input_fn


def run_server(classifier, left_classifier):
    """
    Runs the OSC server for providing predictions.
    :param classifier: Trained tensorflow classifier
    :return:
    """
    server = osc_server.OscServer("127.0.0.1", 54321, classifier, left_classifier)
    server.run_server()


def main(argv):
    """
    Run the training and serving.
    :param argv: Command line arguments.
    :return:
    """
    args = parser.parse_args(argv[1:])

    # Feature columns describe how to use the input.
    my_feature_columns_left = []
    for key in glovedata.FEATURES_LEFT:
        my_feature_columns_left.append(tf.feature_column.numeric_column(key=key))

    my_feature_columns_right = []
    for key in glovedata.FEATURES_RIGHT:
        my_feature_columns_right.append(tf.feature_column.numeric_column(key=key))


    hidden_units = training.hidden_units

    # Wrap the classifier in the FastPredict class to improve prediction speeds.
    classifier = fastpredict.FastPredict(tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns_right,
        hidden_units=hidden_units,
        n_classes=4,
        model_dir=args.model_dir_right), generator_evaluation_fn_right
    )

    left_classifier = fastpredict.FastPredict(tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns_left,
        hidden_units=hidden_units,
        n_classes=4,
        model_dir=args.model_dir_left), generator_evaluation_fn_left
    )

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

    predict_x = tuple([[x] for x in feature_values])
    predictions = classifier.predict(predict_x)

    template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(glovedata.GESTURES[class_id],
                              100 * probability, expec))

    if args.run_server:
        run_server(classifier, left_classifier)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
