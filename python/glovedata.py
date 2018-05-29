import pandas as pd
import tensorflow as tf
import glob
from sklearn.model_selection import train_test_split

GESTURES = ['None', 'Fist', 'Click', 'Point']

FEATURES = ["Human_RightForeArm_Quat_X", "Human_RightForeArm_Quat_Y", "Human_RightForeArm_Quat_Z",
            "Human_RightForeArm_Quat_W", "Human_RightHand_Quat_X", "Human_RightHand_Quat_Y",
            "Human_RightHand_Quat_Z", "Human_RightHand_Quat_W", "Human_RightHandThumb1_Quat_X",
            "Human_RightHandThumb1_Quat_Y", "Human_RightHandThumb1_Quat_Z", "Human_RightHandThumb1_Quat_W",
            "Human_RightHandThumb2_Quat_X", "Human_RightHandThumb2_Quat_Y", "Human_RightHandThumb2_Quat_Z",
            "Human_RightHandThumb2_Quat_W", "Human_RightHandThumb3_Quat_X", "Human_RightHandThumb3_Quat_Y",
            "Human_RightHandThumb3_Quat_Z", "Human_RightHandThumb3_Quat_W", "Human_RightInHandIndex_Quat_X",
            "Human_RightInHandIndex_Quat_Y", "Human_RightInHandIndex_Quat_Z", "Human_RightInHandIndex_Quat_W",
            "Human_RightInHandIndex1_Quat_X", "Human_RightInHandIndex1_Quat_Y", "Human_RightInHandIndex1_Quat_Z",
            "Human_RightInHandIndex1_Quat_W", "Human_RightInHandIndex2_Quat_X", "Human_RightInHandIndex2_Quat_Y",
            "Human_RightInHandIndex2_Quat_Z", "Human_RightInHandIndex2_Quat_W", "Human_RightInHandIndex3_Quat_X",
            "Human_RightInHandIndex3_Quat_Y", "Human_RightInHandIndex3_Quat_Z", "Human_RightInHandIndex3_Quat_W",
            "Human_RightInHandMiddle_Quat_X", "Human_RightInHandMiddle_Quat_Y", "Human_RightInHandMiddle_Quat_Z",
            "Human_RightInHandMiddle_Quat_W", "Human_RightInHandMiddle1_Quat_X", "Human_RightInHandMiddle1_Quat_Y",
            "Human_RightInHandMiddle1_Quat_Z", "Human_RightInHandMiddle1_Quat_W",
            "Human_RightInHandMiddle2_Quat_X", "Human_RightInHandMiddle2_Quat_Y",
            "Human_RightInHandMiddle2_Quat_Z", "Human_RightInHandMiddle2_Quat_W",
            "Human_RightInHandMiddle3_Quat_X", "Human_RightInHandMiddle3_Quat_Y",
            "Human_RightInHandMiddle3_Quat_Z", "Human_RightInHandMiddle3_Quat_W", "Human_RightInHandRing_Quat_X",
            "Human_RightInHandRing_Quat_Y", "Human_RightInHandRing_Quat_Z", "Human_RightInHandRing_Quat_W",
            "Human_RightInHandRing1_Quat_X", "Human_RightInHandRing1_Quat_Y", "Human_RightInHandRing1_Quat_Z",
            "Human_RightInHandRing1_Quat_W", "Human_RightInHandRing2_Quat_X", "Human_RightInHandRing2_Quat_Y",
            "Human_RightInHandRing2_Quat_Z", "Human_RightInHandRing2_Quat_W", "Human_RightInHandRing3_Quat_X",
            "Human_RightInHandRing3_Quat_Y", "Human_RightInHandRing3_Quat_Z", "Human_RightInHandRing3_Quat_W",
            "Human_RightInHandPinky_Quat_X", "Human_RightInHandPinky_Quat_Y", "Human_RightInHandPinky_Quat_Z",
            "Human_RightInHandPinky_Quat_W", "Human_RightInHandPinky1_Quat_X", "Human_RightInHandPinky1_Quat_Y",
            "Human_RightInHandPinky1_Quat_Z", "Human_RightInHandPinky1_Quat_W", "Human_RightInHandPinky2_Quat_X",
            "Human_RightInHandPinky2_Quat_Y", "Human_RightInHandPinky2_Quat_Z", "Human_RightInHandPinky2_Quat_W",
            "Human_RightInHandPinky3_Quat_X", "Human_RightInHandPinky3_Quat_Y", "Human_RightInHandPinky3_Quat_Z",
            "Human_RightInHandPinky3_Quat_W"]


def load_data(path="./data/*.csv", y_name='Gesture'):
    """
    Reads data files in the given path, and returns the dataset as (train_x, train_y), (test_x, test_y).
    :param path: Path to files
    :param y_name: Target feature name
    :return: Dataset
    """

    all_files = glob.glob(path)
    data = pd.concat((pd.read_csv(f, header=0) for f in all_files))
    # convert strings to ints.
    # {'None': 0, 'Fist': 1, 'Click': 2, 'Point': 3}
    mapping = dict(zip(GESTURES, range(len(GESTURES))))
    data.replace({y_name: mapping}, inplace = True)
    data.convert_objects()

    # split up data in a 4/5 split.
    train, test = train_test_split(data, test_size=0.2)
    train_x, train_y = train, train.pop(y_name)
    train_y.convert_objects()
    test_x, test_y = test, test.pop(y_name)
    test_y.convert_objects()
    return (train_x, train_y), (test_x, test_y)


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


