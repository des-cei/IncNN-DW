import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Sequential
import os
import time
import pickle
import multiprocessing as mp
from multiprocessing import Process, get_context

def create_batch(data, i, bs):
    batched_data= data.iloc[(i*bs):(i*bs+bs)]
    array = batched_data.to_numpy()
    reshaped = np.expand_dims(array, axis=0)
    return reshaped

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

def model_train(mt, features, labels):
    with tf.device('/CPU:0'):
        param_unit = [8] #[8, 16, 32, 48]
        batch_size = [10]#[1, 10, 25, 50]
        cv = 5

        resultados = {mt: {}}
        resultados[mt] = {8: {}, 16: {}, 32: {}, 48: {}}
        for units in param_unit:
            resultados[mt][units] = {1: {}, 10: {}, 25: {}, 40: {}}
            for bs in batch_size:
                resultados[mt][units][bs] = {'MAPE-mean-error': {}, 'MAPE-sdv-error': {}, 'MAPE-mean-error_cv': {}, 'Train-time': {}, 'Infer-time': {}}

    for units in param_unit:
        for bs in batch_size:
            print("Modelo - Units - bs: " + mt + " - " + str(units) + " - " + str(bs))
            cv = 10
            infer_time = 0
            train_time = 0
            MAPE_mean = []
            MAPE_std = []

            for j in range(cv):
                buffer_MAPE = []

                model = Sequential([
                            layers.Input(shape=(10, features.shape[1])),
                            layers.LSTM(units),
                            layers.Dense(1, activation='sigmoid')
                        ])
                model.compile(optimizer="adam", loss="mean_absolute_percentage_error")

                if bs == 1:
                    for i in range(labels.size):
                        features_batch = create_batch(features, i, bs)

                        start_time = time.perf_counter()
                        test_loss = float(abs(labels[i]-model.predict_on_batch(features_batch))/labels[i]*100)
                        end_time = time.perf_counter()
                        infer_time = infer_time + end_time-start_time

                        loss = [i, test_loss]
                        buffer_MAPE.append(loss)

                        start_time = time.perf_counter()
                        model.train_on_batch(x=features_batch, y=np.array([[labels[i]]]))
                        end_time = time.perf_counter()
                        train_time = train_time + end_time-start_time

                else:
                    for i in range(labels.size//bs):
                        features_batch = create_batch(features, i, bs)
                        labels_batch = labels.iloc[(i*bs):(i*bs+bs)]
                        labels_mean = float(np.mean(labels_batch))

                        start_time = time.perf_counter()
                        test_loss = float(abs(labels_mean-model.predict_on_batch(features_batch))/labels_mean*100)
                        end_time = time.perf_counter()
                        infer_time = infer_time + end_time-start_time

                        loss = [i, test_loss]
                        buffer_MAPE.append(loss)

                        start_time = time.perf_counter()
                        model.train_on_batch(x=features_batch, y=labels_mean)
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
                resultados[mt][units][bs]['MAPE-mean-error'] = float(np.mean(MAPE_mean))
                resultados[mt][units][bs]['MAPE-sdv-error'] = float(np.mean(MAPE_std))
                resultados[mt][units][bs]['MAPE-mean-error_cv'] = float(np.std(MAPE_mean))
                resultados[mt][units][bs]['Train-time'] = float(train_time*bs/(cv*labels.size))
                resultados[mt][units][bs]['Infer-time'] = float(infer_time/(cv*labels.size))

    with tf.device('/CPU:0'):
        with open("results_LSTM_" + mt + "_bs_10.pkl", "wb") as results_dict:
            pickle.dump(resultados, results_dict)
        results_dict.close()


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
        os.chdir("Results_LSTM/Fractured_dictionaries/Base_LSTM")

        # Parámetros para el multiprocessing
        modelos = ['Top-power', 'Bottom-power', 'Time']


    # Extract features
    features = dataset.drop(["Top power", "Bottom power", "Time"], axis=1)

    # Extract labels
    labels = dataset[["Top power", "Bottom power", "Time"]]
    TP = labels.iloc[:, 0]
    BP = labels.iloc[:, 1]
    Time = labels.iloc[:, 2]

    # Create process
    p1 = ctx.Process(target=model_train, args=(modelos[0], features, TP))
    #p2 = ctx.Process(target=model_train, args=(modelos[1], features, BP))
    #p3 = ctx.Process(target=model_train, args=(modelos[2], features, Time))

    # Start task execution
    p1.start()
    #p2.start()
    #p3.start()

    # Wait for process to complete execution
    p1.join()
    #p2.join()
    #p3.join()


if __name__ == "__main__":
    main()