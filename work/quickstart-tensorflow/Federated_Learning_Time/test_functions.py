import os
import pandas as pd
import keras
from keras import layers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from io import BytesIO
import re

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



def preprocessing(model_type):
    dataset_PATH = "/home/eduardof/work/quickstart-tensorflow"
    os.chdir(dataset_PATH)
    dataset = pd.read_pickle("test_dataset.pkl")

    # Extract features
    features = dataset.drop(["Top power", "Bottom power", "Time"], axis=1)

    # Extract labels
    if model_type == "Top-power":
        labels = dataset.iloc[:, 15]
    elif model_type == "Bottom-power":
        labels = dataset.iloc[:, 16]
    elif model_type == "Time":
        labels = dataset.iloc[:, 17]

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

    #resultados_PATH = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/Testing_dictionaries"
    #os.chdir(resultados_PATH)
    #resultados = pd.read_pickle("resultados_" + model_type + ".pkl")

    return features, labels, model#, resultados



def error_computing(buffer_MAPE, resultados, iteration, server_round, user):
    error_data_k1 = buffer_MAPE[:4020]
    error_values_k1 = [point[1] for point in error_data_k1]
    mean_error_k1 = float(np.mean(error_values_k1))

    error_data_k2 = buffer_MAPE[4020:-3796]
    error_values_k2 = [point[1] for point in error_data_k2]
    mean_error_k2 = float(np.mean(error_values_k2))

    error_data_k3 = buffer_MAPE[-3796:]
    error_values_k3 = [point[1] for point in error_data_k3]
    mean_error_k3 = float(np.mean(error_values_k3))

    error_values = [point[1] for point in buffer_MAPE]
    mean_error = float(np.mean(error_values))

    resultados[iteration][server_round][user]["Kernel_type_1"] = mean_error_k1
    resultados[iteration][server_round][user]["Kernel_type_2"] = mean_error_k2
    resultados[iteration][server_round][user]["Kernel_type_3"] = mean_error_k3
    resultados[iteration][server_round][user]["Full_test"] = mean_error

    return resultados



def model_testing(features, labels, model, resultados, iteration, server_round):
    print("Server round " + str(server_round))
    parameters_PATH = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/parameters_models/server_round_" + str(server_round)
    os.chdir(parameters_PATH)
    param_pkl = "parameters_model_server.pkl"

    params_tensor = pd.read_pickle(param_pkl)
    #parameters = [np.load(BytesIO(blob)) for blob in params_tensor.tensors] --> Not needed as it is saved differently than in clients where it is needed
    model.set_weights(params_tensor)

    buffer_MAPE = []
    for i in range(len(labels)):
        test_loss = float(abs(labels[i]-model.predict_on_batch(features.iloc[[i]]))/labels[i]*100)
        loss = [i, test_loss]
        buffer_MAPE.append(loss)

    resultados = error_computing(buffer_MAPE, resultados, iteration, server_round, "Server_side")
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

        resultados = error_computing(buffer_MAPE, resultados, iteration, server_round, "Client_" + str(j))
        del buffer_MAPE

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
        #line = re.sub(r'(proximal_mu\s*=\s*)[\d.]+', r'\1' + mu_value, line)
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


def modify_seed(file_path: str, line_number: int, iteration: int):
    """Modify or comment the seed line based on the iteration."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Only modify if the line matches the seed pattern
    lines[line_number - 1] = re.sub(r"^(seed\s*=\s*)\d+", rf"\1{iteration}", lines[index])

    # Write back the modified file
    with open(file_path, "w") as f:
        f.writelines(lines)