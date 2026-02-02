import pickle
import os

# Path to get fractured results
os.chdir("Results_LSTM/Fractured_dictionaries/Base_LSTM")

# Levels of the dictionary
model_type = ['Top-power', 'Bottom-power', 'Time']
param_unit = [8, 16, 32, 48]
results = ['MAPE-mean-error', 'MAPE-sdv-error', 'Train-time', 'Infer-time']
results_dummy = ['MAPE_mean_error', 'MAPE_std_error', 'Train_time', 'Infer_time']

# Create the dictionary to store the results
resultados = {'Top-power': {}, 'Bottom-power': {}, 'Time': {}}
for mt in model_type:
    resultados[mt] = {8: {}, 16: {}, 32: {}, 48: {}}
    for units in param_unit:
        resultados[mt][units] = {'MAPE-mean-error': {}, 'MAPE-sdv-error': {}, 'Infer-time': {}, 'Train-time': {}}

for i in range (3):
    # Open each dictionary and copy the data it contans in the new dicionary
    with open('results_LSTM_' + model_type[i] + '.pkl', 'rb') as handle:
        b = pickle.load(handle)
    handle.close()
    for units in param_unit:
        for k in range(4):
            resultados[model_type[i]][units][results[k]] = b[model_type[i]][units][results_dummy[k]]

# Path to store combined results
os.chdir("../../Complete_dictionaries")

with open("LSTM-dictionary.pkl", "wb") as NN_3layers_dict:
    pickle.dump(resultados, NN_3layers_dict)
NN_3layers_dict.close()