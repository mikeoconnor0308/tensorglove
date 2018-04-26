import pandas as pd
import tensorflow as tf
import glob
from sklearn.model_selection import train_test_split


def load_data(path="./data/*.csv", y_name='Gesture'):
    """Returns the dataset as (train_x, train_y), (test_x, test_y)."""

    all_files = glob.glob(path)
    data = pd.concat((pd.read_csv(f, header=0) for f in all_files))

    print(data[y_name])

    # convert strings to ints.
    mapping = {'None': 0, 'Fist': 1, 'Click': 2, 'Point': 3}
    data.replace({y_name: mapping}, inplace = True)
    data.convert_objects()
    print('applied replace')
    print(data[y_name])

    train, test = train_test_split(data, test_size=0.2)
    train_x, train_y = train, train.pop(y_name)
    train_y.convert_objects()
    test_x, test_y = test, test.pop(y_name)
    test_y.convert_objects()
    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
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
