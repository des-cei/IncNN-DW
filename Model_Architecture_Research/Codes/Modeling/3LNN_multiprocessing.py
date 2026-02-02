# CÓDIGO PARA GENERAR REDES NEURONALES DE DOS Y TRES CAPAS Y ORDENARLAS POR RENDIMIENTO

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras import layers, Sequential, initializers
import time
import pickle
import multiprocessing as mp
from multiprocessing import Process, get_context

def rolling(data):
    # Extract the MAPE values from the nested list
    error_data = [point[1] for point in data]
    rolling_mean = np.zeros(len(error_data)-1999)
    rolling_sdv = np.zeros(len(error_data)-1999)

    # Compute the rolling mean for 1000 values
    buffer = error_data[1000:2000]
    rolling_mean[0] = np.mean(buffer)
    rolling_sdv[0] = np.std(buffer)
    j = 0
    for i in range(2000, len(error_data)):
        buffer[j] = error_data[i]
        rolling_mean[i-1999] = np.mean(buffer)
        rolling_sdv[i-1999] = np.std(buffer)
        j += 1
        if j == 1000:
            j = 0
    return rolling_mean, rolling_sdv

# For loop to train the different models, PS Power (TP), PL Power (BP) and execution time
def train_process(mt, L1, features, labels):
    with tf.device('/CPU:0'):
        print("Modelo: " + str(mt) + "  First layer: " + str(L1))
        second_layer = [30,25,20]
        third_layer = [20,15,10]
        batch_size = [50,25,1]

        # Dictionary to store metrics for all synthetic data generation combinations
        resultados = {mt: {}}
        resultados[mt] = {L1: {}}

        resultados[mt][L1] = {30: {}, 25: {}, 20:{}}
        for L2 in second_layer :
            resultados[mt][L1][L2] = {20: {}, 15:{}, 10: {}}
            for L3 in third_layer :
                resultados[mt][L1][L2][L3] = {50: {}, 25: {}, 1: {}}
                for bs in batch_size :
                    resultados[mt][L1][L2][L3][bs] = {'MAPE-mean-error': {}, 'MAPE-sdv-error': {}, 'MAPE-mean-error_cv': {}, 'Train-time': {}, 'Infer-time': {}}

    for L2 in second_layer:
        for L3 in third_layer:
            for bs in batch_size:
                print("Modelo - First Layer - Second Layer - Third layer - bs: " + mt + " - " + str(L1) + " - " + str(L2) + " - " + str(L3) + " - " + str(bs))
                cv = 5
                infer_time = 0
                train_time = 0
                MAPE_mean = []
                MAPE_std = []

                # For loop for cv-fold cross-validation
                for j in range(cv):
                    buffer_MAPE = []

                    model = Sequential([
                        layers.Input(shape=(features.shape[1],)),
                        layers.Dense(L1, activation='relu', kernel_initializer=initializers.RandomNormal(seed=1) , bias_initializer=initializers.zeros()),
                        layers.Dense(L2, activation='relu', kernel_initializer=initializers.RandomNormal(seed=1) , bias_initializer=initializers.zeros()),
                        layers.Dense(L3, activation='relu', kernel_initializer=initializers.RandomNormal(seed=1) , bias_initializer=initializers.zeros()),
                        layers.Dense(1)
                    ])

                    model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

                    if bs == 1:
                        for i in range(labels.size):
                            start_time = time.perf_counter()
                            test_loss = float(abs(labels[i]-model.predict_on_batch(features.iloc[[i]]))/labels[i]*100)
                            end_time = time.perf_counter()
                            infer_time = infer_time + end_time-start_time

                            loss = [i, test_loss]
                            buffer_MAPE.append(loss)

                            start_time = time.perf_counter()
                            model.train_on_batch(x=features.iloc[[i]], y=np.array([[labels[i]]]))
                            end_time = time.perf_counter()
                            train_time = train_time + end_time-start_time

                    else:
                        for i in range(labels.size//bs):
                            features_batch = features.iloc[(i*bs):(i*bs+bs)]
                            labels_batch = labels.iloc[(i*bs):(i*bs+bs)]

                            index_batch = i*bs
                            for k in range(bs):
                                index2 = index_batch + k
                                start_time = time.perf_counter()
                                test_loss = float(abs(labels[index2]-model.predict_on_batch(features.iloc[[index2]]))/labels[index2]*100)
                                end_time = time.perf_counter()
                                infer_time = infer_time + end_time-start_time

                                loss = [i, test_loss]
                                buffer_MAPE.append(loss)

                            start_time = time.perf_counter()
                            model.train_on_batch(x=features_batch, y=np.array([[labels_batch]]))
                            end_time = time.perf_counter()
                            train_time = train_time + end_time-start_time

                        features_batch = features.iloc[(i*bs):]
                        labels_batch = labels.iloc[(i*bs):]

                        index_batch = i*bs
                        for k in range(labels_batch.size):
                            index2 = index_batch + k
                            start_time = time.perf_counter()
                            test_loss = float(abs(labels[index2]-model.predict_on_batch(features.iloc[[index2]]))/labels[index2]*100)
                            end_time = time.perf_counter()
                            infer_time = infer_time + end_time-start_time

                            loss = [i, test_loss]
                            buffer_MAPE.append(loss)

                        start_time = time.perf_counter()
                        model.train_on_batch(x=features_batch, y=np.array([[labels_batch]]))
                        end_time = time.perf_counter()
                        train_time = train_time + end_time-start_time

                    with tf.device('/CPU:0'):
                        rolling_mean, rollind_std = rolling(buffer_MAPE)
                        dummy_mean = np.mean(rolling_mean)
                        MAPE_mean.append(dummy_mean)
                        dummy_std = np.mean(rollind_std)
                        MAPE_std.append(dummy_std)
                        del rolling_mean, rollind_std
                    del buffer_MAPE, model

                with tf.device('/CPU:0'):
                    resultados[mt][L1][L2][L3][bs]['MAPE-mean-error'] = float(np.mean(MAPE_mean))
                    resultados[mt][L1][L2][L3][bs]['MAPE-sdv-error'] = float(np.mean(MAPE_std))
                    resultados[mt][L1][L2][L3][bs]['MAPE-mean-error_cv'] = float(np.std(MAPE_mean))
                    resultados[mt][L1][L2][L3][bs]['Train-time'] = float(train_time*bs/(cv*labels.size))
                    resultados[mt][L1][L2][L3][bs]['Infer-time'] = float(infer_time/(cv*labels.size))

    with tf.device('/CPU:0'):
        with open("NN-3layers-dictionary_" + str(mt) + "_" + str(L1) + ".pkl", "wb") as NN_3layers_dict:
            pickle.dump(resultados, NN_3layers_dict)
        NN_3layers_dict.close()



def main():
    mp.set_start_method('spawn', force=True)
    ctx = get_context("spawn")

    # Confirguramos la GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    with tf.device('/CPU:0'):
        dataset = pd.read_pickle("dataset.pkl")

        # Path to store obtained results
        os.chdir("Results_NN/Fractured_dictionaries/3LNN")

        # Parámetros para el multiprocessing
        modelos = ['Top-power', 'Bottom-power', 'Time']
        first_layer = [35,30,25]

    # Extract features
    features = dataset.drop(["Top power", "Bottom power", "Time"], axis=1)

    # Extract labels
    labels = dataset[["Top power", "Bottom power", "Time"]]
    TP = labels.iloc[:, 0]
    BP = labels.iloc[:, 1]
    Time = labels.iloc[:, 2]

    # Create process
    p1 = ctx.Process(target=train_process, args=(modelos[0], first_layer[0], features, TP))
    p2 = ctx.Process(target=train_process, args=(modelos[0], first_layer[1], features, TP))
    #p3 = ctx.Process(target=train_process, args=(modelos[0], first_layer[2], features, TP))
    #p4 = ctx.Process(target=train_process, args=(modelos[1], first_layer[0], features, BP))
    #p5 = ctx.Process(target=train_process, args=(modelos[1], first_layer[1], features, BP))
    #p6 = ctx.Process(target=train_process, args=(modelos[1], first_layer[2], features, BP))
    #p7 = ctx.Process(target=train_process, args=(modelos[2], first_layer[0], features, Time))
    #p8 = ctx.Process(target=train_process, args=(modelos[2], first_layer[1], features, Time))
    #p9 = ctx.Process(target=train_process, args=(modelos[2], first_layer[2], features, Time))

    # Start task execution
    p1.start()
    p2.start()
    #p3.start()
    #p4.start()
    #p5.start()
    #p6.start()
    #p7.start()
    #p8.start()
    #p9.start()

    # Wait for process to complete execution
    p1.join()
    p2.join()
    #p3.join()
    #p4.join()
    #p5.join()
    #p6.join()
    #p7.join()
    #p8.join()
    #p9.join()


if __name__ == "__main__":
    main()