import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras import layers, Sequential, initializers
import time
import pickle

# Confirguramos la GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Path to dataset
dataset_PATH = "/home/eduardof/work/EduardoFTovar_TFM"
os.chdir(dataset_PATH)
dataset = pd.read_pickle("dataset.pkl")

# Path to store obtained results
STORE_PATH = "/home/eduardof/work/EduardoFTovar_TFM/Results_NN"
os.chdir(STORE_PATH)

modelos = ['Top power', 'Bottom power', 'Time']

# Extract features
features = dataset.drop(["Top power", "Bottom power", "Time"], axis=1)

# Extract labels
labels = dataset[["Top power", "Bottom power", "Time"]]
TP = labels.iloc[:, 0]
BP = labels.iloc[:, 1]
Time = labels.iloc[:, 2]

# Define number of times to perform cross-validation
cv = 5

# Dictionary to store results
resultados = {'Top Power': {}, 'Bottom Power': {}, 'Time': {}}
for modelo in modelos:
    resultados[modelo] = {'Params': {}, 'MAPE-mean-error': {}, 'MAPE-sdv-error': {}, 'Train-time': {}, 'Infer-time': {}}



###############################################################################################################################################
############################################################### Top Power model ###############################################################
###############################################################################################################################################
params_TP = [36, 24, 50] # [First-Layer, Second-Layer, Batch-Size]
params_TP_str = 'First Layer = 36, Second Layer = 24, Batch Size = 50'
infer_time = 0
train_time = 0
MAPE_mean = []
print('Training Best Top Power model: ' + params_TP_str)

for j in range(cv):
    buffer_MAPE = []

    model = Sequential([
        layers.Input(shape=(features.shape[1],)),
        layers.Dense(params_TP[0], activation='relu'),
        layers.Dense(params_TP[1], activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

    for i in range(TP.size//params_TP[2]):
        features_batch = features.iloc[(i*params_TP[2]):(i*params_TP[2]+params_TP[2])]
        labels_batch = TP.iloc[(i*params_TP[2]):(i*params_TP[2]+params_TP[2])]

        index_batch = i*params_TP[2]
        for k in range(params_TP[2]):
            index2 = index_batch + k
            start_time = time.perf_counter()
            test_loss = float(abs(TP[index2]-model.predict_on_batch(features.iloc[[index2]]))/TP[index2]*100)
            end_time = time.perf_counter()
            infer_time = infer_time + end_time-start_time

            loss = [i, test_loss]
            buffer_MAPE.append(loss)

        start_time = time.perf_counter()
        model.train_on_batch(x=features_batch, y=np.array([[labels_batch]]))
        end_time = time.perf_counter()
        train_time = train_time + end_time-start_time

    features_batch = features.iloc[(i*params_TP[2]):]
    labels_batch = TP.iloc[(i*params_TP[2]):]

    index_batch = i*params_TP[2]
    for k in range(labels_batch.size):
        index2 = index_batch + k
        start_time = time.perf_counter()
        test_loss = float(abs(TP[index2]-model.predict_on_batch(features.iloc[[index2]]))/TP[index2]*100)
        end_time = time.perf_counter()
        infer_time = infer_time + end_time-start_time

        loss = [i, test_loss]
        buffer_MAPE.append(loss)

    start_time = time.perf_counter()
    model.train_on_batch(x=features_batch, y=np.array([[labels_batch]]))
    end_time = time.perf_counter()
    train_time = train_time + end_time-start_time

    error_values = [point[1] for point in buffer_MAPE]
    dummy_mean = np.mean(error_values)
    MAPE_mean.append(dummy_mean)

resultados['Top Power']['Params'] = params_TP_str
resultados['Top Power']['MAPE-mean-error'] = float(np.mean(MAPE_mean))
resultados['Top Power']['MAPE-sdv-error'] = float(np.std(MAPE_mean))
resultados['Top Power']['Train-time'] = float(train_time*params_TP[2]/(cv*TP.size)) # Mean train time of a batch
resultados['Top Power']['Infer-time'] = float(infer_time/(cv*TP.size)) # Mean infer time
del MAPE_mean


##############################################################################################################################################
############################################################# Bottom Power model #############################################################
##############################################################################################################################################
params_BP = [24, 8, 25] # [First-Layer, Second-Layer, Batch-Size]
params_BP_str = 'First Layer = 24, Second Layer = 8, Batch Size = 25'
infer_time = 0
train_time = 0
MAPE_mean = []
print('Training Best Bottom Power model: ' + params_BP_str)

for j in range(cv):
    buffer_MAPE = []

    model = Sequential([
        layers.Input(shape=(features.shape[1],)),
        layers.Dense(params_BP[0], activation='relu', kernel_initializer=initializers.RandomNormal(seed=1) , bias_initializer=initializers.zeros()),
        layers.Dense(params_BP[1], activation='relu', kernel_initializer=initializers.RandomNormal(seed=1) , bias_initializer=initializers.zeros()),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

    for i in range(BP.size//params_BP[2]):
        features_batch = features.iloc[(i*params_BP[2]):(i*params_BP[2]+params_BP[2])]
        labels_batch = BP.iloc[(i*params_BP[2]):(i*params_BP[2]+params_BP[2])]

        index_batch = i*params_BP[2]
        for k in range(params_BP[2]):
            index2 = index_batch + k
            start_time = time.perf_counter()
            test_loss = float(abs(BP[index2]-model.predict_on_batch(features.iloc[[index2]]))/BP[index2]*100)
            end_time = time.perf_counter()
            infer_time = infer_time + end_time-start_time

            loss = [i, test_loss]
            buffer_MAPE.append(loss)

        start_time = time.perf_counter()
        model.train_on_batch(x=features_batch, y=np.array([[labels_batch]]))
        end_time = time.perf_counter()
        train_time = train_time + end_time-start_time

    features_batch = features.iloc[(i*params_BP[2]):]
    labels_batch = BP.iloc[(i*params_BP[2]):]

    index_batch = i*params_BP[2]
    for k in range(labels_batch.size):
        index2 = index_batch + k
        start_time = time.perf_counter()
        test_loss = float(abs(BP[index2]-model.predict_on_batch(features.iloc[[index2]]))/BP[index2]*100)
        end_time = time.perf_counter()
        infer_time = infer_time + end_time-start_time

        loss = [i, test_loss]
        buffer_MAPE.append(loss)

    start_time = time.perf_counter()
    model.train_on_batch(x=features_batch, y=np.array([[labels_batch]]))
    end_time = time.perf_counter()
    train_time = train_time + end_time-start_time

    error_values = [point[1] for point in buffer_MAPE]
    dummy_mean = np.mean(error_values)
    MAPE_mean.append(dummy_mean)

resultados['Bottom Power']['Params'] = params_BP_str
resultados['Bottom Power']['MAPE-mean-error'] = float(np.mean(MAPE_mean))
resultados['Bottom Power']['MAPE-sdv-error'] = float(np.std(MAPE_mean))
resultados['Bottom Power']['Train-time'] = float(train_time*params_BP[2]/(cv*BP.size)) # Mean train time
resultados['Bottom Power']['Infer-time'] = float(infer_time/(cv*BP.size)) # Mean infer time
del MAPE_mean



##############################################################################################################################################
############################################################## Time model ####################################################################
##############################################################################################################################################
params_Time = [32, 12, 1] # [First-Layer, Second-Layer, Batch-Size]
params_Time_str = 'First Layer = 32, Second Layer = 12, Batch Size = 1'
infer_time = 0
train_time = 0
MAPE_mean = []
print('Training Best Time model: ' + params_Time_str)

for j in range(cv):
    buffer_MAPE = []

    model = Sequential([
        layers.Input(shape=(features.shape[1],)),
        layers.Dense(params_Time[0], activation='relu'),
        layers.Dense(params_Time[1], activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

    for i in range(Time.size): # Como bs es 1, no necesitamos crear los batches como en el de Top Power
        start_time = time.perf_counter()
        test_loss = float(abs(Time[i]-model.predict_on_batch(features.iloc[[i]]))/Time[i]*100)
        end_time = time.perf_counter()
        infer_time = infer_time + end_time-start_time

        loss = [i, test_loss]
        buffer_MAPE.append(loss)

        start_time = time.perf_counter()
        model.train_on_batch(x=features.iloc[[i]], y=np.array([[Time[i]]]))
        end_time = time.perf_counter()
        train_time = train_time + end_time-start_time


    error_values = [point[1] for point in buffer_MAPE]
    dummy_mean = np.mean(error_values)
    MAPE_mean.append(dummy_mean)

resultados['Time']['Params'] = params_Time_str
resultados['Time']['MAPE-mean-error'] = float(np.mean(MAPE_mean))
resultados['Time']['MAPE-sdv-error'] = float(np.std(MAPE_mean))
resultados['Time']['Train-time'] = float(train_time*params_Time[2]/(cv*Time.size)) # Mean train time
resultados['Time']['Infer-time'] = float(infer_time/(cv*Time.size)) # Mean infer time

with open("Best_models_results_cv_" + str(cv) + ".pkl", "wb") as Best_models_dict:
    pickle.dump(resultados, Best_models_dict)
Best_models_dict.close()