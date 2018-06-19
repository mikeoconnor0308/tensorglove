import pandas as pd
import tensorflow as tf
import glob
from sklearn.model_selection import train_test_split

GESTURES = ['None', 'Fist', 'Click', 'Point']

FEATURES_RIGHT = ["Human_RightForeArm_Quat_X", "Human_RightForeArm_Quat_Y", "Human_RightForeArm_Quat_Z",
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
                  "Human_RightInHandMiddle_Quat_W", "Human_RightInHandMiddle1_Quat_X",
                  "Human_RightInHandMiddle1_Quat_Y",
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

FEATURES_LEFT = ["Human_LeftForeArm_Quat_X", "Human_LeftForeArm_Quat_Y", "Human_LeftForeArm_Quat_Z",
                 "Human_LeftForeArm_Quat_W", "Human_LeftHand_Quat_X", "Human_LeftHand_Quat_Y", "Human_LeftHand_Quat_Z",
                 "Human_LeftHand_Quat_W", "Human_LeftHandThumb1_Quat_X", "Human_LeftHandThumb1_Quat_Y",
                 "Human_LeftHandThumb1_Quat_Z", "Human_LeftHandThumb1_Quat_W", "Human_LeftHandThumb2_Quat_X",
                 "Human_LeftHandThumb2_Quat_Y", "Human_LeftHandThumb2_Quat_Z", "Human_LeftHandThumb2_Quat_W",
                 "Human_LeftHandThumb3_Quat_X", "Human_LeftHandThumb3_Quat_Y", "Human_LeftHandThumb3_Quat_Z",
                 "Human_LeftHandThumb3_Quat_W", "Human_LeftInHandIndex_Quat_X", "Human_LeftInHandIndex_Quat_Y",
                 "Human_LeftInHandIndex_Quat_Z", "Human_LeftInHandIndex_Quat_W", "Human_LeftInHandIndex1_Quat_X",
                 "Human_LeftInHandIndex1_Quat_Y", "Human_LeftInHandIndex1_Quat_Z", "Human_LeftInHandIndex1_Quat_W",
                 "Human_LeftInHandIndex2_Quat_X", "Human_LeftInHandIndex2_Quat_Y", "Human_LeftInHandIndex2_Quat_Z",
                 "Human_LeftInHandIndex2_Quat_W", "Human_LeftInHandIndex3_Quat_X", "Human_LeftInHandIndex3_Quat_Y",
                 "Human_LeftInHandIndex3_Quat_Z", "Human_LeftInHandIndex3_Quat_W", "Human_LeftInHandMiddle_Quat_X",
                 "Human_LeftInHandMiddle_Quat_Y", "Human_LeftInHandMiddle_Quat_Z", "Human_LeftInHandMiddle_Quat_W",
                 "Human_LeftInHandMiddle1_Quat_X", "Human_LeftInHandMiddle1_Quat_Y", "Human_LeftInHandMiddle1_Quat_Z",
                 "Human_LeftInHandMiddle1_Quat_W", "Human_LeftInHandMiddle2_Quat_X", "Human_LeftInHandMiddle2_Quat_Y",
                 "Human_LeftInHandMiddle2_Quat_Z", "Human_LeftInHandMiddle2_Quat_W", "Human_LeftInHandMiddle3_Quat_X",
                 "Human_LeftInHandMiddle3_Quat_Y", "Human_LeftInHandMiddle3_Quat_Z", "Human_LeftInHandMiddle3_Quat_W",
                 "Human_LeftInHandRing_Quat_X", "Human_LeftInHandRing_Quat_Y", "Human_LeftInHandRing_Quat_Z",
                 "Human_LeftInHandRing_Quat_W", "Human_LeftInHandRing1_Quat_X", "Human_LeftInHandRing1_Quat_Y",
                 "Human_LeftInHandRing1_Quat_Z", "Human_LeftInHandRing1_Quat_W", "Human_LeftInHandRing2_Quat_X",
                 "Human_LeftInHandRing2_Quat_Y", "Human_LeftInHandRing2_Quat_Z", "Human_LeftInHandRing2_Quat_W",
                 "Human_LeftInHandRing3_Quat_X", "Human_LeftInHandRing3_Quat_Y", "Human_LeftInHandRing3_Quat_Z",
                 "Human_LeftInHandRing3_Quat_W", "Human_LeftInHandPinky_Quat_X", "Human_LeftInHandPinky_Quat_Y",
                 "Human_LeftInHandPinky_Quat_Z", "Human_LeftInHandPinky_Quat_W", "Human_LeftInHandPinky1_Quat_X",
                 "Human_LeftInHandPinky1_Quat_Y", "Human_LeftInHandPinky1_Quat_Z", "Human_LeftInHandPinky1_Quat_W",
                 "Human_LeftInHandPinky2_Quat_X", "Human_LeftInHandPinky2_Quat_Y", "Human_LeftInHandPinky2_Quat_Z",
                 "Human_LeftInHandPinky2_Quat_W", "Human_LeftInHandPinky3_Quat_X", "Human_LeftInHandPinky3_Quat_Y",
                 "Human_LeftInHandPinky3_Quat_Z", "Human_LeftInHandPinky3_Quat_W"]



def load_data(right_path="./data/right_*.csv", left_path="./data/left_*.csv", y_name='Gesture'):
    """
    Reads data files in the given path for left and right hands, and returns the dataset as (train_left_x,
    train_left__y), (test_left_x, test_left_y), (train_right_x, train_right__y), (test_right_x, test_right_y).
    :param
    path: Path to files :param y_name: Target feature name :return: Dataset
    """

    (train_left_x, train_left_y), (test_left_x, test_left_y) = _load_data(left_path)
    (train_right_x, train_right_y), (test_right_x, test_right_y) = _load_data(right_path)

    return (train_left_x, train_left_y), (test_left_x, test_left_y), (train_right_x, train_right_y), (
    test_right_x, test_right_y)


def _load_data(path, y_name='Gesture'):
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
    data.replace({y_name: mapping}, inplace=True)
    data.convert_objects()

    # split up data in a 4/5 split.
    train, test = train_test_split(data, test_size=0.2)
    train_x, train_y = train, train.pop(y_name)
    train_y.convert_objects()
    test_x, test_y = test, test.pop(y_name)
    test_y.convert_objects()
    return (train_x, train_y), (test_x, test_y)
