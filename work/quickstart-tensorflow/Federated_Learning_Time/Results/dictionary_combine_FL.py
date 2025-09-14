import pickle
import os

# Path to get fractured results
DATA_PATH = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/Testing_dictionaries"
os.chdir(DATA_PATH)

# Levels of the dictionary
model_type = ['Top-power', 'Bottom-power', 'Time']
fed_strat = ["Bulyan", "Fault-Tolerant-FedAvg", "FedAdagrad", "FedAdam", "FedAvg", "FedAvgM", "FedMedian", "FedOpt", "FedProx_1", "FedProx_0.1", "FedProx_0.01", "FedProx_0.001", "FedTrimmedAvg", "FedYogi", "Krum"]
server_rounds = [1,2,3]
testers = ["Server_side", "Client_1", "Client_2", "Client_3"]
result_type = ["Kernel_type_1", "Kernel_type_2", "Kernel_type_3", "Full_test"]
results = ["Mean", "sdv", "Time"]

# Create the dictionary to store the results
resultados = {'Top-power': {}, 'Bottom-power': {}, 'Time': {}}
for mt in model_type:
    resultados[mt] = {"Bulyan": {}, "Fault-Tolerant-FedAvg": {}, "FedAdagrad": {}, "FedAdam": {}, "FedAvg": {}, "FedAvgM": {}, "FedMedian": {}, "FedOpt": {}, "FedProx_1": {}, "FedProx_0.1": {}, "FedProx_0.01": {}, "FedProx_0.001": {}, "FedTrimmedAvg": {}, "FedYogi": {}, "Krum": {}}
    for Strat in fed_strat:
        resultados[mt][Strat] = {1: {}, 2:{}, 3: {},}
        for sr in server_rounds:
            resultados[mt][Strat][sr] = {"Server_side": {}, "Client_1": {}, "Client_2": {}, "Client_3": {}}
            for tester in testers:
                resultados[mt][Strat][sr][tester] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}
                for rt in result_type:
                    resultados[mt][Strat][sr][tester][rt] = {"Mean": {}, "sdv": {}, "Time": {}}

for i in range (3):
    # Open each dictionary and copy the data it contans in the new dicionary
    with open('resultados_' + model_type[i] + '_processed.pkl', 'rb') as handle:
        b = pickle.load(handle)
    handle.close()
    for Strat in fed_strat:
        for sr in server_rounds:
            for tester in testers:
                for rt in result_type:
                    for result in results:
                        resultados[model_type[i]][Strat][sr][tester][rt][result] = b[Strat][sr][tester][rt][result]

# Path to store combined results
#STORE_PATH = "/home/eduardof/work/EduardoFTovar_TFM/Results_NN/Complete_dictionaries"
#os.chdir(STORE_PATH)

with open("Resultados_FL_procesados.pkl", "wb") as NN_2layers_dict:
    pickle.dump(resultados, NN_2layers_dict)
NN_2layers_dict.close()