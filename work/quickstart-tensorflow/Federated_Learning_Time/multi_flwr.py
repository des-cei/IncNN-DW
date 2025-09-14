import subprocess
import os
from test_functions import preprocessing, model_testing, replace_strategy_in_file, modify_proximal_mu, modify_seed
import pickle
import numpy as np
import pandas as pd
from flwr.common.logger import log
from logging import WARN
import time

# Variables
seed = "Seed_1" #
fed_strat = ["Bulyan", "Fault-Tolerant-FedAvg", "FedAdagrad", "FedAdam", "FedAvg", "FedAvgM", "FedMedian", "FedOpt", "FedProx_1", "FedProx_0.1", "FedProx_0.01", "FedProx_0.001", "FedTrimmedAvg", "FedYogi", "Krum"]
iteration = [1, 2, 3, 4, 5] # --> serves as cross valitation
server_rounds = [1,2,3]
testers = ["Server_side", "Client_1", "Client_2", "Client_3"]
result_type = ["Kernel_type_1", "Kernel_type_2", "Kernel_type_3", "Full_test"]

# Create dictionary with final results
results_PATH = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/Testing_dictionaries"
os.chdir(results_PATH)
resultados = {"Bulyan": {}, "Fault-Tolerant-FedAvg": {}, "FedAdagrad": {}, "FedAdam": {}, "FedAvg": {}, "FedAvgM": {}, "FedMedian": {}, "FedOpt": {}, "FedProx_1": {}, "FedProx_0.1": {}, "FedProx_0.01": {}, "FedProx_0.001": {}, "FedTrimmedAvg": {}, "FedYogi": {}, "Krum": {}}
for Strat in fed_strat:
    resultados[Strat] = {1: {}, 2:{}, 3: {},}
    for sr in server_rounds:
        resultados[Strat][sr] = {"Server_side": {}, "Client_1": {}, "Client_2": {}, "Client_3": {}}
        for tester in testers:
            resultados[Strat][sr][tester] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}
            for rt in result_type:
                resultados[Strat][sr][tester][rt] = {"Mean": {}, "sdv": {}, "Time": {}}

# Create dummy dictionary for getting iter results to then compute mean and sdv
dummy = {1: {}, 2:{}, 3: {}, 4:{}, 5: {}}
for iter in iteration:
    dummy[iter] = {1: {}, 2:{}, 3: {}}
    for sr in server_rounds:
        dummy[iter][sr] = {"Server_side": {}, "Client_1": {}, "Client_2": {}, "Client_3": {}}
        for tester in testers:
            dummy[iter][sr][tester] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}



# Define model type and extract data testing models
model_type = "Top-power" #"Top-power"   "Bottom-power"   "Time"
features, labels, model = preprocessing(model_type)

# Variables for running flwr command
working_directory = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time"
command = ["flwr", "run"]

# Variables to modify strategy for flwr
file_to_edit = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Program/server_app.py"
target_line = 66
target_line_FedProx = 73
seed_file = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Program/task.py"
seed_line = 18



for Strat in fed_strat:
    if Strat == "Fault-Tolerant-FedAvg":
        replace_strategy_in_file(file_to_edit, target_line, "FaultTolerantFedAvg")
    elif Strat == "FedProx_1":
        replace_strategy_in_file(file_to_edit, target_line, "FedProx")
        modify_proximal_mu(file_to_edit, target_line_FedProx, Strat)
    elif Strat == "FedProx_0.1":
        modify_proximal_mu(file_to_edit, target_line_FedProx, Strat)
    elif Strat == "FedProx_0.01":
        modify_proximal_mu(file_to_edit, target_line_FedProx, Strat)
    elif Strat == "FedProx_0.001":
        modify_proximal_mu(file_to_edit, target_line_FedProx, Strat)
    elif Strat == "FedTrimmedAvg":
        modify_proximal_mu(file_to_edit, target_line_FedProx, Strat)
        replace_strategy_in_file(file_to_edit, target_line, Strat)
    else:
        replace_strategy_in_file(file_to_edit, target_line, Strat)

    mean_time = 0
    for iter in iteration:
        modify_seed(seed_file, seed_line, iter)
        log(WARN, f"{Strat} iteration {iter}")
        start_time = time.perf_counter()
        subprocess.run(command, cwd=working_directory, check=True)
        final_time = time.perf_counter()
        mean_time += final_time - start_time
        for sr in server_rounds:
            dummy = model_testing(features, labels, model, dummy, iter, sr)
    mean_time = mean_time/len(iteration)

    for sr in server_rounds:
        for tester in testers:
            for rt in result_type:
                buffer_array = []
                for iter in iteration:
                    buffer_array.append(dummy[iter][sr][tester][rt])
                mean = float(np.mean(buffer_array))
                sdv = float(np.std(buffer_array))
                resultados[Strat][sr][tester][rt]["Mean"] = mean
                resultados[Strat][sr][tester][rt]["sdv"] = sdv
                resultados[Strat][sr][tester][rt]["Time"] = mean_time


    results_PATH = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/Testing_dictionaries"
    os.chdir(results_PATH)
    with open("resultados_Top-power_processed.pkl", "wb") as handler:
        pickle.dump(resultados, handler)
    handler.close()

