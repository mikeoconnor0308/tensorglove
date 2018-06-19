"""
Script for training the DNNClassifier.
"""
import tensorflow as tf
import glovedata
from glovedata import FEATURES_LEFT
from glovedata import FEATURES_RIGHT
import sys

hidden_units = [18, 20]
batch_size = 100
train_steps = 10000

def train_input_fn(features, labels, batch_size):
    """
    An input function for training the neural network.
    """
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """
    An input function for evaluation or prediction
    """
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def train(training_x, training_y,  test_x, test_y, model_dir, feature_keys):
    # train the left hand
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in training_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=hidden_units,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001
        ),
        n_classes=4,
        model_dir=model_dir)

    print("Training model")
    # Train the Model.
    classifier.train(
        input_fn=lambda: train_input_fn(training_x, training_y,
                                        batch_size),
        steps=train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_x, test_y,
                                       batch_size))

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
    predict_x = dict(zip(feature_keys, [[x] for x in feature_values]))

    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_x,
                                       labels=None,
                                       batch_size=batch_size))

    template = '\nPrediction is "{}" ({:.1f}%), expected "{}"'

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(glovedata.GESTURES[class_id],
                              100 * probability, expec))


def main(argv):
    """
    Run the training
    :param argv: Command line arguments.
    :return:
    """

    # Fetch the data
    (train_left_x_all, train_left_y_all), (test_left_x, test_left_y), (train_right_x_all, train_right_y_all), (test_right_x, test_right_y) = glovedata.load_data()
    print('training left hand.')
    train(train_left_x_all, train_left_y_all, test_left_x, test_left_y, "model_left", FEATURES_LEFT)
    print('training right hand.')
    train(train_right_x_all, train_right_y_all, test_right_x, test_right_y, "model_right", FEATURES_RIGHT)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run(main)

    main(sys.argv)
