import pickle
import os

# Path to get fractured results
os.chdir("Results_NN/Fractured_dictionaries/2LNN")

# Levels of the dictionary
model_type = ['Top-power', 'Bottom-power', 'Time']
first_layer = [36, 32, 28, 24, 20]
second_layer =  [24, 20, 16, 12, 8]
batch_size = [50,25,10,1]
results = ['MAPE-mean-error', 'MAPE-sdv-error', 'MAPE-mean-error_cv', 'MAPE-sdv-error_cv', 'Train-time', 'Infer-time']

# Create the dictionary to store the results
resultados = {'Top-power': {}, 'Bottom-power': {}, 'Time': {}}
for mt in model_type:
    resultados[mt] = {36: {}, 32: {}, 28: {}, 24: {}, 20: {}}
    for L1 in first_layer:
        resultados[mt][L1] = {24: {}, 20: {}, 16: {}, 12: {}, 8: {}}
        for L2 in second_layer :
                resultados[mt][L1][L2] = {50: {}, 25: {}, 10:{}, 1: {}}
                for bs in batch_size :
                    resultados[mt][L1][L2][bs] = {'MAPE-mean-error': {}, 'MAPE-sdv-error': {}, 'MAPE-mean-error_cv': {}, 'MAPE-sdv-error_cv': {}, 'Train-time': {}, 'Infer-time': {}}

for i in range (3):
    # Open each dictionary and copy the data it contans in the new dicionary
    with open('NN-2layers-dictionary_' + model_type[i] + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    handle.close()
    for L1 in first_layer:
        for L2 in second_layer:
            for bs in batch_size:
                for result in results:
                    if result == 'Train-time':
                        resultados[model_type[i]][L1][L2][bs][result] = b[model_type[i]][L1][L2][bs][result]*98525/bs
                    else:
                        resultados[model_type[i]][L1][L2][bs][result] = b[model_type[i]][L1][L2][bs][result]

# Path to store combined results
os.chdir("../../Complete_dictionaries")

with open("NN-2layers-dictionary.pkl", "wb") as NN_2layers_dict:
    pickle.dump(resultados, NN_2layers_dict)
NN_2layers_dict.close()