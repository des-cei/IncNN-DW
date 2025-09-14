import os
import pandas as pd
import numpy as np
import pickle

results_PATH = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/Testing_dictionaries"
os.chdir(results_PATH)
resultados_dummy = pd.read_pickle("resultados_Time.pkl")

seeds = ["Seed_1", "Seed_2", "Seed_3"]
fed_strat_org = ["Bulyan", "Fault-Tolerant-FedAvg", "FedAdagrad", "FedAdam", "FedAvg", "FedAvgM", "FedMedian", "FedOpt", "FedProx", "FedTrimmeredAvg", "FedYogi", "Krum"]
fed_strat_new = ["Bulyan", "Fault-Tolerant-FedAvg", "FedAdagrad", "FedAdam", "FedAvg", "FedAvgM", "FedMedian", "FedOpt", "FedProx_1", "FedProx_0.1", "FedProx_0.01", "FedProx_0.001", "FedTrimmeredAvg", "FedYogi", "Krum"]
learning_rate = [0.01, 0.005, 0.001, 0.0005]
iteration = [1, 2, 3, 4, 5]
server_rounds = [1,2,3]
testers = ["Server_side", "Client_1", "Client_2", "Client_3"]
result_type = ["Kernel_type_1", "Kernel_type_2", "Kernel_type_3", "Full_test"]

resultados = {"Seed_1": {}, "Seed_2": {}, "Seed_3": {}}
for seed in seeds:
    resultados[seed] = {"Bulyan": {}, "Fault-Tolerant-FedAvg": {}, "FedAdagrad": {}, "FedAdam": {}, "FedAvg": {}, "FedAvgM": {}, "FedMedian": {}, "FedOpt": {}, "FedProx_1": {}, "FedProx_0.1": {}, "FedProx_0.01": {}, "FedProx_0.001": {}, "FedTrimmeredAvg": {}, "FedYogi": {}, "Krum": {}}
    for Strat in fed_strat_new:
        resultados[seed][Strat] = {1: {}, 2:{}, 3: {},}
        for sr in server_rounds:
            resultados[seed][Strat][sr] = {"Server_side": {}, "Client_1": {}, "Client_2": {}, "Client_3": {}}
            for tester in testers:
                resultados[seed][Strat][sr][tester] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}
                for rt in result_type:
                    resultados[seed][Strat][sr][tester][rt] = {"Mean": {}, "Sdv": {}}

for Strat in fed_strat_org:
    if Strat == "FedProx":
        for sr in server_rounds:
            for tester in testers:
                    for rt in result_type:
                        buffer_array_1 = []
                        buffer_array_2 = []
                        buffer_array_3 = []
                        buffer_array_4 = []
                        for iter in iteration:
                            buffer_array_1.append(resultados_dummy[Strat][0.01][iter][sr][tester][rt])
                            buffer_array_2.append(resultados_dummy[Strat][0.005][iter][sr][tester][rt])
                            buffer_array_3.append(resultados_dummy[Strat][0.001][iter][sr][tester][rt])
                            buffer_array_4.append(resultados_dummy[Strat][0.0005][iter][sr][tester][rt])
                        resultados["Seed_1"]["FedProx_1"][sr][tester][rt]["Mean"] = float(np.mean(buffer_array_1))
                        resultados["Seed_1"]["FedProx_1"][sr][tester][rt]["Sdv"] = float(np.std(buffer_array_1))
                        resultados["Seed_1"]["FedProx_0.1"][sr][tester][rt]["Mean"] = float(np.mean(buffer_array_2))
                        resultados["Seed_1"]["FedProx_0.1"][sr][tester][rt]["Sdv"] = float(np.std(buffer_array_2))
                        resultados["Seed_1"]["FedProx_0.01"][sr][tester][rt]["Mean"] = float(np.mean(buffer_array_3))
                        resultados["Seed_1"]["FedProx_0.01"][sr][tester][rt]["Sdv"] = float(np.std(buffer_array_3))
                        resultados["Seed_1"]["FedProx_0.001"][sr][tester][rt]["Mean"] = float(np.mean(buffer_array_4))
                        resultados["Seed_1"]["FedProx_0.001"][sr][tester][rt]["Sdv"] = float(np.std(buffer_array_4))

    else:
        for sr in server_rounds:
            for tester in testers:
                    for rt in result_type:
                        buffer_array = []
                        for iter in iteration:
                            buffer_array.append(resultados_dummy[Strat][0.001][iter][sr][tester][rt])
                        resultados["Seed_1"][Strat][sr][tester][rt]["Mean"] = float(np.mean(buffer_array))
                        resultados["Seed_1"][Strat][sr][tester][rt]["Sdv"] = float(np.std(buffer_array))

with open("resultados_Time_processed.pkl", "wb") as handler:
    pickle.dump(resultados, handler)
handler.close()

