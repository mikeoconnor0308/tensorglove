import argparse
import tensorflow as tf
import glovedata

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=5, type=int, help='batch size')
parser.add_argument('--train_steps', default=800, type=int,
                    help='number of training steps')

feature_names = ['Human_RightForeArm_Quat_X','Human_RightForeArm_Quat_Y','Human_RightForeArm_Quat_Z','Human_RightForeArm_Quat_W','Human_RightHand_Quat_X','Human_RightHand_Quat_Y','Human_RightHand_Quat_Z','Human_RightHand_Quat_W','Human_RightHandThumb1_Quat_X','Human_RightHandThumb1_Quat_Y','Human_RightHandThumb1_Quat_Z','Human_RightHandThumb1_Quat_W','Human_RightHandThumb2_Quat_X','Human_RightHandThumb2_Quat_Y','Human_RightHandThumb2_Quat_Z','Human_RightHandThumb2_Quat_W','Human_RightHandThumb3_Quat_X','Human_RightHandThumb3_Quat_Y','Human_RightHandThumb3_Quat_Z','Human_RightHandThumb3_Quat_W','Human_RightInHandIndex_Quat_X','Human_RightInHandIndex_Quat_Y','Human_RightInHandIndex_Quat_Z','Human_RightInHandIndex_Quat_W','Human_RightInHandIndex1_Quat_X','Human_RightInHandIndex1_Quat_Y','Human_RightInHandIndex1_Quat_Z','Human_RightInHandIndex1_Quat_W','Human_RightInHandIndex2_Quat_X','Human_RightInHandIndex2_Quat_Y','Human_RightInHandIndex2_Quat_Z','Human_RightInHandIndex2_Quat_W','Human_RightInHandIndex3_Quat_X','Human_RightInHandIndex3_Quat_Y','Human_RightInHandIndex3_Quat_Z','Human_RightInHandIndex3_Quat_W','Human_RightInHandMiddle_Quat_X','Human_RightInHandMiddle_Quat_Y','Human_RightInHandMiddle_Quat_Z','Human_RightInHandMiddle_Quat_W','Human_RightInHandMiddle1_Quat_X','Human_RightInHandMiddle1_Quat_Y','Human_RightInHandMiddle1_Quat_Z','Human_RightInHandMiddle1_Quat_W','Human_RightInHandMiddle2_Quat_X','Human_RightInHandMiddle2_Quat_Y','Human_RightInHandMiddle2_Quat_Z','Human_RightInHandMiddle2_Quat_W','Human_RightInHandMiddle3_Quat_X','Human_RightInHandMiddle3_Quat_Y','Human_RightInHandMiddle3_Quat_Z','Human_RightInHandMiddle3_Quat_W','Human_RightInHandRing_Quat_X','Human_RightInHandRing_Quat_Y','Human_RightInHandRing_Quat_Z','Human_RightInHandRing_Quat_W','Human_RightInHandRing1_Quat_X','Human_RightInHandRing1_Quat_Y','Human_RightInHandRing1_Quat_Z','Human_RightInHandRing1_Quat_W','Human_RightInHandRing2_Quat_X','Human_RightInHandRing2_Quat_Y','Human_RightInHandRing2_Quat_Z','Human_RightInHandRing2_Quat_W','Human_RightInHandRing3_Quat_X','Human_RightInHandRing3_Quat_Y','Human_RightInHandRing3_Quat_Z','Human_RightInHandRing3_Quat_W','Human_RightInHandPinky_Quat_X','Human_RightInHandPinky_Quat_Y','Human_RightInHandPinky_Quat_Z','Human_RightInHandPinky_Quat_W','Human_RightInHandPinky1_Quat_X','Human_RightInHandPinky1_Quat_Y','Human_RightInHandPinky1_Quat_Z','Human_RightInHandPinky1_Quat_W','Human_RightInHandPinky2_Quat_X','Human_RightInHandPinky2_Quat_Y','Human_RightInHandPinky2_Quat_Z','Human_RightInHandPinky2_Quat_W','Human_RightInHandPinky3_Quat_X','Human_RightInHandPinky3_Quat_Y','Human_RightInHandPinky3_Quat_Z','Human_RightInHandPinky3_Quat_W']

def serving_input_receiver_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat in feature_names:
        inputs[feat] = tf.placeholder(shape=[None], dtype='float32')

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in inputs.items()
    }

    return tf.estimator.export.ServingInputReceiver(features,
                                                    inputs)


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = glovedata.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10],
        # The model must choose between 4 classes.
        n_classes=4,
        model_dir="model")

    # Train the Model.
    classifier.train(
        input_fn=lambda: glovedata.train_input_fn(train_x, train_y,
                                                  args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda: glovedata.eval_input_fn(test_x, test_y,
                                                 args.batch_size))

    export_dir = classifier.export_savedmodel(
    export_dir_base="model_export",
    serving_input_receiver_fn=serving_input_receiver_fn)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
