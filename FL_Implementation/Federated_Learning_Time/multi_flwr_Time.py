import subprocess
import os
from test_functions import preprocessing_time, model_testing, replace_strategy_in_file, modify_proximal_mu, modify_k_fold, prepare_test_database
import pickle
import numpy as np
import pandas as pd
from flwr.common.logger import log
from logging import INFO

# Variables
fed_strat = ["Bulyan", "Fault-Tolerant-FedAvg", "FedAdagrad", "FedAdam", "FedAvg", "FedAvgM", "FedMedian", "FedOpt", "FedProx_1", "FedProx_0.1", "FedProx_0.01", "FedProx_0.001", "FedTrimmedAvg", "FedYogi", "Krum"]
iteration = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # --> serves as cross valitation
testers = ["Server_side", "Client_1", "Client_2", "Client_3"]
result_type = ["Kernel_type_1", "Kernel_type_2", "Kernel_type_3", "Full_test"]

# Create dictionary with final results
resultados = {"Bulyan": {}, "Fault-Tolerant-FedAvg": {}, "FedAdagrad": {}, "FedAdam": {}, "FedAvg": {}, "FedAvgM": {}, "FedMedian": {}, "FedOpt": {}, "FedProx_1": {}, "FedProx_0.1": {}, "FedProx_0.01": {}, "FedProx_0.001": {}, "FedTrimmedAvg": {}, "FedYogi": {}, "Krum": {}}
for Strat in fed_strat:
    resultados[Strat] = {"Server_side": {}, "Client_1": {}, "Client_2": {}, "Client_3": {}}
    for tester in testers:
        resultados[Strat][tester] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}
        for rt in result_type:
            resultados[Strat][tester][rt] = {"Mean": {}, "sdv": {}, "Time": {}}

# Create dummy dictionary for getting iter results to then compute mean and sdv
dummy = {0: {}, 1: {}, 2:{}, 3: {}, 4:{}, 5: {}, 6: {}, 7:{}, 8: {}, 9:{}}
for iter in iteration:
    dummy[iter] = {"Server_side": {}, "Client_1": {}, "Client_2": {}, "Client_3": {}}
    for tester in testers:
        dummy[iter][tester] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}

# Define model type and extract data testing models
model_type = "Time"
model = preprocessing_time()

# Variables for running flwr command
cwd = os.getcwd()
working_directory = cwd + "/Federated_Learning_Time"
command = ["flwr", "run"]

# Variables to modify strategy for flwr
strategy_file = "/Federated_Learning_Time/Program/server_app.py"
strategy_target_line = 48
FedProx_target_line = 54
k_fold_file = "/Federated_Learning_Time/Program/task.py"
k_fold_target_lines = [51, 81]

for Strat in fed_strat:
    if Strat == "Fault-Tolerant-FedAvg":
        replace_strategy_in_file(strategy_file, strategy_target_line, "FaultTolerantFedAvg")
    elif Strat == "FedProx_1":
        replace_strategy_in_file(strategy_file, strategy_target_line, "FedProx")
        modify_proximal_mu(strategy_file, FedProx_target_line, Strat)
    elif Strat == "FedProx_0.1":
        modify_proximal_mu(strategy_file, FedProx_target_line, Strat)
    elif Strat == "FedProx_0.01":
        modify_proximal_mu(strategy_file, FedProx_target_line, Strat)
    elif Strat == "FedProx_0.001":
        modify_proximal_mu(strategy_file, FedProx_target_line, Strat)
    elif Strat == "FedTrimmedAvg":
        modify_proximal_mu(strategy_file, FedProx_target_line, Strat)
        replace_strategy_in_file(strategy_file, strategy_target_line, Strat)
    else:
        replace_strategy_in_file(strategy_file, strategy_target_line, Strat)

    mean_time = 0
    for iter in iteration:
        modify_k_fold(k_fold_file, k_fold_target_lines, iter)
        log(INFO, f"{Strat} iteration {iter}")
        subprocess.run(command, cwd=working_directory, check=True)
        mean_time+= pd.read_pickle(os.path.join(cwd, "Results", "provisionary_parameters_models", "server_round_1", "aggregate_time.pkl"))
        features, labels = prepare_test_database(iter, model_type)
        dummy = model_testing(features, labels, model, dummy, iter)
    mean_time = mean_time/len(iteration)

    for tester in testers:
        for rt in result_type:
            buffer_array = []
            for iter in iteration:
                buffer_array.append(dummy[iter][tester][rt])
            mean = float(np.mean(buffer_array))
            sdv = float(np.std(buffer_array))
            resultados[Strat][tester][rt]["Mean"] = mean
            resultados[Strat][tester][rt]["sdv"] = sdv
            resultados[Strat][tester][rt]["Time"] = mean_time

    path = os.path.join(
        cwd,
        "Results",
        "Testing_dictionaries",
        f"resultados_{model_type}_processed.pkl"
    )
    with open(path, "wb") as handler:
        pickle.dump(resultados, handler)
    handler.close()