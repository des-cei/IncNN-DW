import os
from test_functions import preprocessing, model_testing
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd

model_type = "Time" #"Top-power"   "Bottom-power"   "Time"
features, labels, model, resultados = preprocessing(model_type)

# ["Bulyan", "Fault-Tolerant-FedAvg", "FedAdagrad", "FedAdam", "FedAvg", "FedAvgM", "FedMedian", "FedOpt", "FedProx", "FedTrimmeredAvg", "FedYogi", "Krum"]
fed_strat = "Bulyan"
learning_rate = 0.001 #[0.01, 0.005, 0.001, 0.0005]
iteration = 1 #[1, 2, 3, 4, 5] --> serves as cross valitation

"""
server_rounds = [1,2,3]
testers = ["Server_side", "Clien_1", "Client_2", "Client_3"]

resultados = {"Bulyan": {}, "Fault-Tolerant-FedAvg": {}, "FedAdagrad": {}, "FedAdam": {}, "FedAvg": {}, "FedAvgM": {}, "FedMedian": {}, "FedOpt": {}, "FedProx": {}, "FedTrimmeredAvg": {}, "FedYogi": {}, "Krum": {}}
for Strat in fed_strat:
    resultados[Strat] = {0.01: {}, 0.005: {}, 0.001: {}, 0.005: {}}
    for lr in learning_rate:
        resultados[Strat][lr] = {1: {}, 2:{}, 3: {}, 4:{}, 5: {}}
        for iter in iteration:
            resultados[Strat][lr][iter] = {1: {}, 2:{}, 3: {},}
            for sr in server_rounds:
                resultados[Strat][lr][iter][sr] = {"Server_side": {}, "Client_1": {}, "Client_2": {}, "Client_3": {}}
                for tester in testers:
                    resultados[Strat][lr][iter][sr][tester] = {"Kernel_type_1": {}, "Kernel_type_2": {}, "Kernel_type_3": {}, "Full_test": {}}
"""
"""
parameters_PATH = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/parameters_models/server_round_0"
os.chdir(parameters_PATH)
param_pkl = "parameters_model_server.pkl"

params_tensor = pd.read_pickle(param_pkl)
#parameters = [np.load(BytesIO(blob)) for blob in params_tensor.tensors] --> Not needed as it is saved differently than in clients where it is needed
model.set_weights(params_tensor)

buffer_MAPE = []
for i in range(len(labels)):
    test_loss = float(abs(labels[i]-model.predict_on_batch(features.iloc[[i]]))/labels[i]*100)
    loss = [i, test_loss]
    buffer_MAPE.append(loss)

error_data_k1 = buffer_MAPE[:4020]
error_values_k1 = [point[1] for point in error_data_k1]
mean_error_k1 = float(np.mean(error_values_k1))

error_data_k2 = buffer_MAPE[4020:-3796]
error_values_k2 = [point[1] for point in error_data_k2]
mean_error_k2 = float(np.mean(error_values_k2))

error_data_k3 = buffer_MAPE[-3796:]
error_values_k3 = [point[1] for point in error_data_k3]
mean_error_k3 = float(np.mean(error_values_k3))

error_values = [point[1] for point in buffer_MAPE]
mean_error = float(np.mean(error_values))
"""

for server_round in range(1,4):
    resultados = model_testing(features, labels, model, resultados, fed_strat, learning_rate, iteration, server_round)

resultados_PATH = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/Testing_dictionaries"
os.chdir(resultados_PATH)
with open("resultados_" + model_type + ".pkl", "wb") as handler:
    pickle.dump(resultados, handler)
handler.close()

"""
#parameters = params_tensor.tensors
#parameters = [tensor.numpy() for tensor in params_tensor.tensors]
#parameters = params_tensor.numpy()
#parameters = list(params_tensor.tensors)
parameters_PATH = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/parameters_models/server_round_1"
os.chdir(parameters_PATH)
param_pkl = "parameters_model_client_0.pkl"

params_tensor = pd.read_pickle(param_pkl)
parameters = [np.load(BytesIO(blob)) for blob in params_tensor.tensors]
#if isinstance(parameters, np.ndarray) and parameters.dtype == object:
#    parameters = list(parameters)
model.set_weights(parameters)

buffer_MAPE = []
for i in range(len(Time)):
    test = float(model.predict_on_batch(features.iloc[[i]]))
    test_loss = float(abs(Time[i]-model.predict_on_batch(features.iloc[[i]]))/Time[i]*100)
    loss = [i, test_loss]
    buffer_MAPE.append(loss)

error_values = [point[1] for point in buffer_MAPE]
dummy_mean = float(np.mean(error_values))
print("Mean error of the test on client 1: " + str(dummy_mean))

#errors1 = rolling_mean(buffer_MAPE)
#errors1 = [point[1] for point in buffer_MAPE]

# Extract features
#features = dataset.drop(["Top power", "Bottom power", "Time"], axis=1)

# Extract labels time labels
#Time = dataset.iloc[:, 17]

#client_process(features, Time, "parameters_model_0.pkl", 1)
#client_process(features, Time, "parameters_model_1.pkl", 2)
#client_process(features, Time, "parameters_model_2.pkl", 3)

#clients_process(features, Time)
"""