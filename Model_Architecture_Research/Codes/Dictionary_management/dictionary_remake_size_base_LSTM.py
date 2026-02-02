import pickle
import os

# Path to read and store results
os.chdir("Results_LSTM/Complete_dictionaries")

# Levels of the dictionary
model_type = ['Top-power', 'Bottom-power', 'Time']
param_unit = [8, 16, 32, 48]
batch_size = [1, 10, 25, 50]
results = ['MAPE-mean-error', 'MAPE-sdv-error', 'Train-time', 'Infer-time']

# Create the dictionary to store the results
resultados = {'Top-power': {}, 'Bottom-power': {}, 'Time': {}}

# Open each dictionary and copy the data it contans in the new dicionary
with open('LSTM-dictionary.pkl', 'rb') as handle:
    b = pickle.load(handle)
handle.close()

number_models = len(param_unit) * len(batch_size)

for mt in model_type:
    resultados[mt] = {str(k): {} for k in range(number_models)}
    for k in range(number_models):
        resultados[mt][str(k)] = {'MAPE-mean-error': {}, 'MAPE-sdv-error': {}, 'Train-time': {}, 'Infer-time': {}, 'Model_name': {}}

for mt in model_type:
    i = 0
    for unit in param_unit:
        for bs in batch_size:
            model_name = 'Nº of units: ' + str(unit)
            for result in results:
                resultados[mt][str(i)][result] = b[mt][unit][bs][result]
            resultados[mt][str(i)]["Model_name"] = model_name
            i += 1


with open("LSTM-dictionary_pareto.pkl", "wb") as NN_3layers_dict:
    pickle.dump(resultados, NN_3layers_dict)
NN_3layers_dict.close()