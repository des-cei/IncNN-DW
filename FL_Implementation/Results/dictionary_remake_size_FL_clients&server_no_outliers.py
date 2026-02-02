import pickle
import os

# Path to read and store results
os.chdir("Results/Testing_dictionaries")

# Levels of the dictionary
model_type = ['Top-power', 'Bottom-power', 'Time']
fed_strat = ["Bulyan", "Fault-Tolerant-FedAvg", "FedAdagrad", "FedAdam", "FedAvg", "FedAvgM", "FedMedian", "FedOpt", "FedProx_1", "FedProx_0.1", "FedProx_0.01", "FedProx_0.001", "FedTrimmedAvg", "FedYogi", "Krum"]
testers = ["Server_side", "Client_1", "Client_2", "Client_3"]
result_type = ["Kernel_type_1", "Kernel_type_2", "Kernel_type_3", "Full_test"]
results = ["Mean", "sdv", "Time"]

with open('Resultados_FL_procesados.pkl', 'rb') as handle:
    b = pickle.load(handle)
handle.close()

# Create the dictionary to store the results
resultados = {'Top-power': {}, 'Bottom-power': {}, 'Time': {}}

Top_Power_outliers = 3
resultados['Top-power'] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}
for rt in result_type:
    resultados['Top-power'][rt] = {str(k): {} for k in range((len(fed_strat) - Top_Power_outliers) * len(testers))}
    for k in range((len(fed_strat) - Top_Power_outliers) * len(testers)):
        resultados['Top-power'][rt][str(k)] = {'Mean': {}, 'sdv': {}, "Time": {}, 'Model_name': {}, 'User': {}}

Bottom_Power_outliers = 3
resultados['Bottom-power'] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}
for rt in result_type:
    resultados['Bottom-power'][rt] = {str(k): {} for k in range((len(fed_strat) - Bottom_Power_outliers) * len(testers))}
    for k in range((len(fed_strat) - Bottom_Power_outliers) * len(testers)):
        resultados['Bottom-power'][rt][str(k)] = {'Mean': {}, 'sdv': {}, "Time": {}, 'Model_name': {}, 'User': {}}

Time_Power_outliers = 2
resultados['Time'] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}
for rt in result_type:
    resultados['Time'][rt] = {str(k): {} for k in range((len(fed_strat) - Time_Power_outliers) * len(testers))}
    for k in range((len(fed_strat) - Time_Power_outliers) * len(testers)):
        resultados['Time'][rt][str(k)] = {'Mean': {}, 'sdv': {}, "Time": {}, 'Model_name': {}, 'User': {}}

for rt in result_type:
    i = 0
    for Strat in fed_strat:
        for tester in testers:
            if ((Strat != "FedAdam") and (Strat != "FedAdagrad") and (Strat != "FedYogi")):
                model_name = 'Strat: ' + Strat + ', tester: ' + tester + ', server round: 1'
                for result in results:
                    resultados['Top-power'][rt][str(i)][result] = b['Top-power'][Strat][tester][rt][result]
                resultados['Top-power'][rt][str(i)]["Model_name"] = model_name
                resultados['Top-power'][rt][str(i)]['User'] = tester
                i += 1

for rt in result_type:
    i = 0
    for Strat in fed_strat:
        for tester in testers:
            if ((Strat != "FedAdam") and (Strat != "FedAdagrad") and (Strat != "FedYogi")):
                model_name = 'Strat: ' + Strat + ', tester: ' + tester + ', server round: 1'
                for result in results:
                    resultados['Bottom-power'][rt][str(i)][result] = b['Bottom-power'][Strat][tester][rt][result]
                resultados['Bottom-power'][rt][str(i)]["Model_name"] = model_name
                resultados['Bottom-power'][rt][str(i)]['User'] = tester
                i += 1

for rt in result_type:
    i = 0
    for Strat in fed_strat:
        for tester in testers:
            if ((Strat != "Krum") and (Strat != "FedYogi")):
                model_name = 'Strat: ' + Strat + ', tester: ' + tester + ', server round: 1'
                for result in results:
                    resultados['Time'][rt][str(i)][result] = b['Time'][Strat][tester][rt][result]
                resultados['Time'][rt][str(i)]["Model_name"] = model_name
                resultados['Time'][rt][str(i)]['User'] = tester
                i += 1


with open("Resultados_FL_sr1_all_outlier_free.pkl", "wb") as new_dict:
    pickle.dump(resultados, new_dict)
new_dict.close()