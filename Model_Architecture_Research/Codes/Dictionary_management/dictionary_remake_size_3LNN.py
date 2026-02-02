import pickle
import os

# Path to read and store results
os.chdir("Results_NN/Complete_dictionaries")

# Levels of the dictionary
model_type = ['Top-power', 'Bottom-power', 'Time']
first_layer = [35, 30, 25]
second_layer = [30,25,20]
third_layer = [20,15,10]
batch_size = [50,25,1]
results = ['MAPE-mean-error', 'MAPE-sdv-error', 'MAPE-mean-error_cv', 'MAPE-sdv-error_cv', 'Train-time', 'Infer-time']

# Create the dictionary to store the results
resultados = {'Top-power': {}, 'Bottom-power': {}, 'Time': {}}

# Open each dictionary and copy the data it contans in the new dicionary
with open('NN-3layers-dictionary.pkl', 'rb') as handle:
    a = pickle.load(handle)
handle.close()

number_models = len(first_layer) * len(second_layer) * len(third_layer) * len(batch_size)

for mt in model_type:
    resultados[mt] = {str(k): {} for k in range(number_models)}
    for k in range(number_models):
        resultados[mt][str(k)] = {'MAPE-mean-error': {}, 'MAPE-sdv-error': {}, 'MAPE-mean-error_cv': {}, 'MAPE-sdv-error_cv': {}, 'Train-time': {}, 'Infer-time': {}}

for mt in model_type:
    i = 0
    for L1 in first_layer:
        for L2 in second_layer:
            for L3 in third_layer:
                for bs in batch_size:
                    model_name = 'FL: ' + str(L1) + ', SL: ' + str(L2) + ', TL: ' + str(L3) + ', BS: ' + str(bs)
                    for result in results:
                        resultados[mt][str(i)][result] = a[mt][L1][L2][L3][bs][result]
                    resultados[mt][str(i)]["Model_name"] = model_name
                    i += 1


with open("NN-3layers-dictionary_pareto.pkl", "wb") as NN_3layers_dict:
    pickle.dump(resultados, NN_3layers_dict)
NN_3layers_dict.close()