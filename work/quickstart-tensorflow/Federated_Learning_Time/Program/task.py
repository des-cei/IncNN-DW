"""tfexample: A Flower / TensorFlow app."""

import os

import keras
import pandas as pd
from keras import layers, initializers
from sklearn.model_selection import train_test_split
from flwr.common.logger import log
from logging import INFO

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(params = [28, 0.2, 12], input_shape = 15, learning_rate: float = 0.001):
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    seed = 1
    model = keras.Sequential(
        [
            layers.Input(shape=(input_shape,)),
            layers.Dense(36, activation="relu", kernel_initializer=initializers.RandomNormal(seed=seed) , bias_initializer=initializers.zeros()),
            layers.Dense(24, activation="relu", kernel_initializer=initializers.RandomNormal(seed=seed) , bias_initializer=initializers.zeros()),
            layers.Dense(1, kernel_initializer=initializers.RandomNormal(seed=seed) , bias_initializer=initializers.zeros()),
        ]
    )
    log(INFO, f"Checkpoint saved")

    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mean_absolute_percentage_error",
        metrics=["mean_absolute_percentage_error"],
    )
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Path to dataset
    dataset_PATH = "/home/eduardof/work/quickstart-tensorflow"
    os.chdir(dataset_PATH)
    if partition_id == 0:
        dataset = pd.read_pickle("dataset_client_1.pkl")
    elif partition_id == 1:
        dataset = pd.read_pickle("dataset_client_2.pkl")
    elif partition_id == 2:
        dataset = pd.read_pickle("dataset_client_3.pkl")


    # Extract features
    features = dataset.drop(["Top power", "Bottom power", "Time"], axis=1)

    # Extract labels time labels
    Time = dataset.iloc[:,15]

    x_train, x_val, y_train, y_val = train_test_split(features, Time, test_size = 0.2, random_state = 50)
    """
    del dataset, features, Time

    dataset = pd.read_pickle("test_dataset.pkl")

    # Extract features
    x_test = dataset.drop(["Top power", "Bottom power", "Time"], axis=1)

    # Extract labels time labels
    y_test = dataset.iloc[:,17]
    """

    return x_train, y_train, x_val, y_val#, x_test, y_test
