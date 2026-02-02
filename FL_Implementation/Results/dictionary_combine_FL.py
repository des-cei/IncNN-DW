import pickle
import os

# Path to get fractured results
os.chdir("Results/Testing_dictionaries")

# Levels of the dictionary
model_type = ['Top-power', 'Bottom-power', 'Time']
fed_strat = ["Bulyan", "Fault-Tolerant-FedAvg", "FedAdagrad", "FedAdam", "FedAvg", "FedAvgM", "FedMedian", "FedOpt", "FedProx_1", "FedProx_0.1", "FedProx_0.01", "FedProx_0.001", "FedTrimmedAvg", "FedYogi", "Krum"]
testers = ["Server_side", "Client_1", "Client_2", "Client_3"]
result_type = ["Kernel_type_1", "Kernel_type_2", "Kernel_type_3", "Full_test"]
results = ["Mean", "sdv", "Time"]

# Create the dictionary to store the results
resultados = {'Top-power': {}, 'Bottom-power': {}, 'Time': {}}
for mt in model_type:
    resultados[mt] = {"Bulyan": {}, "Fault-Tolerant-FedAvg": {}, "FedAdagrad": {}, "FedAdam": {}, "FedAvg": {}, "FedAvgM": {}, "FedMedian": {}, "FedOpt": {}, "FedProx_1": {}, "FedProx_0.1": {}, "FedProx_0.01": {}, "FedProx_0.001": {}, "FedTrimmedAvg": {}, "FedYogi": {}, "Krum": {}}
    for Strat in fed_strat:
        resultados[mt][Strat] = {"Server_side": {}, "Client_1": {}, "Client_2": {}, "Client_3": {}}
        for tester in testers:
            resultados[mt][Strat][tester] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}
            for rt in result_type:
                resultados[mt][Strat][tester][rt] = {"Mean": {}, "sdv": {}, "Time": {}}

for mt in model_type:
    # Open each dictionary and copy the data it contans in the new dicionary
    with open('resultados_' + mt + '_processed.pkl', 'rb') as handle:
        b = pickle.load(handle)
    handle.close()
    for Strat in fed_strat:
        for tester in testers:
            for rt in result_type:
                for result in results:
                    resultados[mt][Strat][tester][rt][result] = b[Strat][tester][rt][result]

with open("Resultados_FL_procesados.pkl", "wb") as combined_dict:
    pickle.dump(resultados, combined_dict)
combined_dict.close()