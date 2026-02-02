import pickle
import os

# Path to get fractured results
os.chdir("Results_NN/Fractured_dictionaries/3LNN")

# Levels of the dictionary
model_type = ['Top-power', 'Bottom-power', 'Time']
first_layer = [35, 30, 25]
second_layer = [30,25,20]
third_layer = [20,15,10]
batch_size = [50,25,1]
results = ['MAPE-mean-error', 'MAPE-sdv-error', 'MAPE-mean-error_cv', 'MAPE-sdv-error_cv', 'Train-time', 'Infer-time']

# Create the dictionary to store the results
resultados = {'Top-power': {}, 'Bottom-power': {}, 'Time': {}}
for mt in model_type:
    resultados[mt] = {35: {}, 30: {}, 25: {}}
    for L1 in first_layer:
        resultados[mt][L1] = {30: {}, 25: {}, 20:{}}
        for L2 in second_layer:
            resultados[mt][L1][L2] = {20: {}, 15:{}, 10: {}}
            for L3 in third_layer:
                resultados[mt][L1][L2][L3] = {50: {}, 25: {}, 1: {}}
                for bs in batch_size:
                    resultados[mt][L1][L2][L3][bs] = {'MAPE-mean-error': {}, 'MAPE-sdv-error': {}, 'MAPE-mean-error_cv': {}, 'MAPE-sdv-error_cv': {}, 'Train-time': {}, 'Infer-time': {}}

for i in range (3):
    for k in range (3):
        # Open each dictionary and copy the data it contans in the new dicionary
        with open('NN-3layers-dictionary_' + model_type[i] + '_' + str(first_layer[k]) + '.pkl', 'rb') as handle:
            b = pickle.load(handle)
        handle.close()

        for L2 in second_layer:
            for L3 in third_layer:
                for bs in batch_size:
                    for result in results:
                        if result == 'Train-time':
                            resultados[model_type[i]][first_layer[k]][L2][L3][bs][result] = b[model_type[i]][first_layer[k]][L2][L3][bs][result]*98525/bs
                        else:
                            resultados[model_type[i]][first_layer[k]][L2][L3][bs][result] = b[model_type[i]][first_layer[k]][L2][L3][bs][result]

# Path to store combined results
os.chdir("../../Complete_dictionaries")

with open("NN-3layers-dictionary.pkl", "wb") as NN_3layers_dict:
    pickle.dump(resultados, NN_3layers_dict)
NN_3layers_dict.close()