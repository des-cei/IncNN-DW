import os
import pandas as pd
import keras
from keras import layers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from io import BytesIO
import re

"""
def rolling_mean(data, sdv = None):
    # Extract the MAPE values from the nested list
    rolling_mean = [point[1] for point in data]
    rolling_sdv = []

    # Compute the rolling mean for 100 values
    mean_buffer = rolling_mean[:100]
    rolling_mean[99] = np.mean(mean_buffer)
    j = 0
    if sdv != None:
        for i in range(100, rolling_mean.size):
            mean_buffer[j] = rolling_mean[i]
            rolling_mean[i] = np.mean(mean_buffer)
            rolling_sdv[i-100] = np.std(mean_buffer)
            j += 1
            if j == 100:
                j = 0
        return rolling_mean, rolling_sdv

    else:
        for i in range(100, len(rolling_mean)):
            mean_buffer[j] = rolling_mean[i]
            rolling_mean[i] = np.mean(mean_buffer)
            j += 1
            if j == 100:
                j = 0
        return rolling_mean
"""


def preprocessing_top_power():
    model = keras.Sequential(
        [
            layers.Input(shape=(15,)),
            layers.Dense(36, activation="relu"),
            layers.Dense(24, activation="relu"),
            layers.Dense(1),
        ]
    )

    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        optimizer=optimizer,
        loss="mean_absolute_percentage_error",
        metrics=["mean_absolute_percentage_error"],
    )
    return model



def preprocessing_bottom_power():

    model = keras.Sequential(
        [
            layers.Input(shape=(15,)),
            layers.Dense(24, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(1),
        ]
    )

    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        optimizer=optimizer,
        loss="mean_absolute_percentage_error",
        metrics=["mean_absolute_percentage_error"],
    )
    return model



def preprocessing_time():
    model = keras.Sequential(
        [
            layers.Input(shape=(15,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(12, activation="relu"),
            layers.Dense(1),
        ]
    )

    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        optimizer=optimizer,
        loss="mean_absolute_percentage_error",
        metrics=["mean_absolute_percentage_error"],
    )
    return model



def error_computing(buffer_MAPE, resultados, iteration, user):
    error_data_k1 = buffer_MAPE[:2680]
    error_values_k1 = [point[1] for point in error_data_k1]
    mean_error_k1 = float(np.mean(error_values_k1))

    error_data_k2 = buffer_MAPE[2680:-2531]
    error_values_k2 = [point[1] for point in error_data_k2]
    mean_error_k2 = float(np.mean(error_values_k2))

    error_data_k3 = buffer_MAPE[-2531:]
    error_values_k3 = [point[1] for point in error_data_k3]
    mean_error_k3 = float(np.mean(error_values_k3))

    error_values = [point[1] for point in buffer_MAPE]
    mean_error = float(np.mean(error_values))

    resultados[iteration][user]["Kernel_type_1"] = mean_error_k1
    resultados[iteration][user]["Kernel_type_2"] = mean_error_k2
    resultados[iteration][user]["Kernel_type_3"] = mean_error_k3
    resultados[iteration][user]["Full_test"] = mean_error

    return resultados



def model_testing(features, labels, model, resultados, iteration):
    os.chdir("Results/provisionary_parameters_models/server_round_1")
    param_pkl = "parameters_model_server.pkl"

    params_tensor = pd.read_pickle(param_pkl)
    model.set_weights(params_tensor)

    buffer_MAPE = []
    for i in range(len(labels)):
        test_loss = float(abs(labels[i]-model.predict_on_batch(features.iloc[[i]]))/labels[i]*100)
        loss = [i, test_loss]
        buffer_MAPE.append(loss)

    resultados = error_computing(buffer_MAPE, resultados, iteration, "Server_side")
    del buffer_MAPE

    for j in range(1,4):
        param_pkl = "parameters_model_client_" + str(j) + ".pkl"
        params_tensor = pd.read_pickle(param_pkl)
        parameters = [np.load(BytesIO(blob)) for blob in params_tensor.tensors]

        model.set_weights(parameters)

        buffer_MAPE = []
        for i in range(len(labels)):
            test_loss = float(abs(labels[i]-model.predict_on_batch(features.iloc[[i]]))/labels[i]*100)
            loss = [i, test_loss]
            buffer_MAPE.append(loss)

        resultados = error_computing(buffer_MAPE, resultados, iteration, "Client_" + str(j))
        del buffer_MAPE

    os.chdir("../../..")

    return resultados



def replace_strategy_in_file(file_path: str, line_number: int, new_strategy: str):
    """Replace the strategy class name in a specific line of a Python file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Replace only the targeted line
    if line_number - 1 < len(lines):
        lines[line_number - 1] = replace_strategy_name(lines[line_number - 1], new_strategy)

        with open(file_path, 'w') as file:
            file.writelines(lines)
        print(f"Line {line_number} updated to use '{new_strategy}' strategy.")
    else:
        print(f"Line {line_number} does not exist in the file.")



def replace_strategy_name(line: str, new_strategy: str) -> str:
    """Replace the strategy class name in the line with the new one."""
    return re.sub(r'(strategy\s*=\s*strategy\s*=\s*)\w+\(', r'\1' + new_strategy + '(', line)



def modify_proximal_mu(file_path: str, line_number: int, strat_value: str):
    """Modify or comment the proximal_mu line based on the strategy."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if line_number - 1 >= len(lines):
        print(f"Line {line_number} does not exist.")
        return

    line = lines[line_number - 1]

    # Case 1: Strategy requires FedProx with a specific value
    match = re.match(r"FedProx_(\d*\.?\d+)", strat_value)
    if match:
        mu_value = match.group(1)
        # Uncomment the line
        line = re.sub(r'^(\s*)#(proximal_mu\s*=)', r'\1\2', line)
        # Set the correct value
        line = re.sub(r'(proximal_mu\s*=\s*)[\d.]+', lambda m: m.group(1) + mu_value, line)

    # Case 2: Strategy is FedTrimmedAvg → comment the line
    elif strat_value == "FedTrimmedAvg":
        # Only comment if not already commented
        if not re.match(r'^\s*#', line):
            line = re.sub(r'^(\s*)', r'\1#', line)

    # Update the line in the file
    lines[line_number - 1] = line

    with open(file_path, 'w') as file:
        file.writelines(lines)


def modify_k_fold(file_path: str, line_numbers: int, new_value: int):
    """
    Modifies n lines of a file with the format 'selected = <digit>' by replacing the digit with new_value (0–9).
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if not (0 <= new_value <= 9):
        raise ValueError("new_value must be a single digit from 0 to 9.")

    for i in range(len(line_numbers)):
        if line_numbers[i] < 1 or line_numbers[i] > len(lines):
            raise IndexError(f"Line {line_numbers[i]} is out of range for '{file_path}'.")

        target_line = lines[line_numbers[i] - 1]

        modified_line = re.sub(
            r'(selected\s*=\s*)\d',
            lambda m: f"{m.group(1)}{new_value}",
            target_line
        )

        lines[line_numbers[i] - 1] = modified_line

    with open(file_path, 'w') as f:
        f.writelines(lines)



def prepare_test_database(iter, model):
    dataset1 = pd.read_pickle("dataset_client_1.pkl")
    dataset2 = pd.read_pickle("dataset_client_2.pkl")
    dataset3 = pd.read_pickle("dataset_client_3.pkl")

    # Shuffle dataset
    dataset1 = dataset1.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset2 = dataset2.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset3 = dataset3.sample(frac=1, random_state=42).reset_index(drop=True)

    # Prepare folds
    folds1 = np.array_split(dataset1, 10)
    folds2 = np.array_split(dataset2, 10)
    folds3 = np.array_split(dataset3, 10)

    test_df = pd.concat([folds1[iter], folds2[iter], folds3[iter]], axis=0, ignore_index=True)

    # Extract features
    features = test_df.drop(["Top power", "Bottom power", "Time"], axis=1)

    # Extract labels
    if model == "Top-power":
        labels = test_df.iloc[:, 15]
    elif model == "Bottom-power":
        labels = test_df.iloc[:, 16]
    elif model == "Time":
        labels = test_df.iloc[:, 17]

    return features, labels