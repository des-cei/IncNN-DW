import pickle
import os

# Path to read and store results
STORE_PATH = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/Testing_dictionaries"
os.chdir(STORE_PATH)

# Levels of the dictionary
model_type = ['Top-power', 'Bottom-power', 'Time']
fed_strat = ["Bulyan", "Fault-Tolerant-FedAvg", "FedAdagrad", "FedAdam", "FedAvg", "FedAvgM", "FedMedian", "FedOpt", "FedProx_1", "FedProx_0.1", "FedProx_0.01", "FedProx_0.001", "FedTrimmedAvg", "FedYogi", "Krum"]
server_rounds = [1,2,3]
testers = ["Server_side", "Client_1", "Client_2", "Client_3"]
result_type = ["Kernel_type_1", "Kernel_type_2", "Kernel_type_3", "Full_test"]
results = ["Mean", "sdv", "Time"]

# Open each dictionary and copy the data it contans in the new dicionary
with open('Resultados_FL_procesados.pkl', 'rb') as handle:
    b = pickle.load(handle)
handle.close()

# Create the dictionary to store the results
resultados = {'Top-power': {}, 'Bottom-power': {}, 'Time': {}}
for mt in model_type:
    resultados[mt] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}
    for rt in result_type:
        resultados[mt][rt] = {str(k): {} for k in range(15)}
        for k in range(15):
            resultados[mt][rt][str(k)] = {'Mean': {}, 'sdv': {}, "Time": {}, 'Model_name': {}}

for mt in model_type:
    for rt in result_type:
        i = 0
        for Strat in fed_strat:
            model_name = 'Strat: ' + Strat + ', tester: Server, server round: 1'
            for result in results:
                resultados[mt][rt][str(i)][result] = b[mt][Strat][1]["Server_side"][rt][result]
            resultados[mt][rt][str(i)]["Model_name"] = model_name
            i += 1


with open("Resultados_FL_sr1_server.pkl", "wb") as NN_2layers_dict:
    pickle.dump(resultados, NN_2layers_dict)
NN_2layers_dict.close()