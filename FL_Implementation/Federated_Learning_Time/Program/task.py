"""tfexample: A Flower / TensorFlow app."""
import os
import numpy as np
import keras
import pandas as pd
from keras import layers, initializers

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_model(learning_rate: float = 0.001, input_shape = 15):
    seed = 1
    model = keras.Sequential(
        [
            layers.Input(shape=(input_shape,)),
            layers.Dense(32, activation="relu", kernel_initializer=initializers.RandomNormal(seed=seed) , bias_initializer=initializers.zeros()),
            layers.Dense(12, activation="relu", kernel_initializer=initializers.RandomNormal(seed=seed) , bias_initializer=initializers.zeros()),
            layers.Dense(1, kernel_initializer=initializers.RandomNormal(seed=seed) , bias_initializer=initializers.zeros()),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mean_absolute_percentage_error",
        metrics=["mean_absolute_percentage_error"],
    )
    return model

fds = None  # Cache FederatedDataset

def load_data_server():
    dataset_path = os.path.dirname(os.getcwd())
    dataset_1 = pd.read_pickle(os.path.join(dataset_path, "dataset_client_1.pkl"))
    dataset_2 = pd.read_pickle(os.path.join(dataset_path, "dataset_client_2.pkl"))
    dataset_3 = pd.read_pickle(os.path.join(dataset_path, "dataset_client_3.pkl"))

    # Shuffle dataset
    dataset_1 = dataset_1.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset_2 = dataset_2.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset_3 = dataset_3.sample(frac=1, random_state=42).reset_index(drop=True)

    # Prepare folds
    folds_1 = np.array_split(dataset_1, 10)
    folds_2 = np.array_split(dataset_2, 10)
    folds_3 = np.array_split(dataset_3, 10)

    # Choose which fold you want
    selected = 1

    test_df = pd.concat([folds_1[selected], folds_2[selected], folds_3[selected]], axis=0, ignore_index=True)

    # Extract features
    features = test_df.drop(["Top power", "Bottom power", "Time"], axis=1)

    # Extract labels
    labels = test_df.iloc[:, 17]

    return features, labels

def load_data_clients(partition_id):
    dataset_path = os.path.dirname(os.getcwd())
    if partition_id == 0:
        dataset = pd.read_pickle(os.path.join(dataset_path, "dataset_client_1.pkl"))
    elif partition_id == 1:
        dataset = pd.read_pickle(os.path.join(dataset_path, "dataset_client_2.pkl"))
    elif partition_id == 2:
        dataset = pd.read_pickle(os.path.join(dataset_path, "dataset_client_3.pkl"))

    # Shuffle dataset
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    # Prepare folds
    folds = np.array_split(dataset, 10)

    # Choose which fold you want
    selected = 1

    test_df = folds[selected]
    train_df = pd.concat([f for i, f in enumerate(folds) if i != selected],
                        ignore_index=True)

    # Separate features and labels
    x_train = train_df.drop(["Top power", "Bottom power", "Time"], axis=1)
    x_val  = test_df.drop(["Top power", "Bottom power", "Time"], axis=1)
    y_train = train_df.iloc[:,17]
    y_val  = test_df.iloc[:,17]

    return x_train, y_train, x_val, y_val