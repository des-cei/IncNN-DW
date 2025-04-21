import numpy as np
import pandas as pd
import os
from tensorflow import keras
from keras import layers
import time
import matplotlib as mpl
import matplotlib.pyplot as plt


def features_labels_accommodation(features, labels):
    """Perform accomodation on features and labels. Type casting..."""
    
    features["user"] = float(features["user"])
    features["kernel"] = float(features["kernel"])
    features["idle"] = float(features["idle"])

    features["Main"] = int(features["Main"])
    features["aes"] = int(features["aes"])
    features["bulk"] = int(features["bulk"])
    features["crs"] = int(features["crs"])
    features["kmp"] = int(features["kmp"])
    features["knn"] = int(features["knn"])
    features["merge"] = int(features["merge"])
    features["nw"] = int(features["nw"])
    features["queue"] = int(features["queue"])
    features["stencil2d"] = int(features["stencil2d"])
    features["stencil3d"] = int(features["stencil3d"])
    features["strided"] = int(features["strided"])

    # Get each model label
    labels = [float(labels[key]) for key in labels]

    return features, labels



def train_cont_model_2layer(features, labels, param):
    # Name of the model to train
    print("\nContinuous training of a two layer neural network: First layer = " + str(param[0]) + " neurones, Dropout = " + str(param[1]) + ", Second layer = " + str(param[2]) + " neurones")
    model_name = "FL = " + str(param[0]) + ", D = " + str(param[1]) + ", SL = " + str(param[2])
    name_list = ["Model:", model_name]
     
    # Nested lists with results to return
    results = []
    results.append(name_list)
    buffer_MAPE = []

    start_time = time.perf_counter()

    # Create the 2 layer neural network
    model = keras.Sequential([
            layers.Input(shape=(features.shape[1],)),
            layers.Dense(param[0], activation="relu"),
            layers.Dropout(param[1]),
            layers.Dense(param[2], activation="relu"),
            layers.Dense(1)
        ])
    model.compile(optimizer="adam", loss="mean_absolute_percentage_error")
    
    # Train the network and save the errors
    for i in range(labels.size):
        test_loss = float(abs(labels[i]-model.predict_on_batch(features.iloc[[i]]))/labels[i]*100)
        loss = [i, test_loss]
        buffer_MAPE.append(loss)
        model.train_on_batch(x=features.iloc[[i]], y=np.array([[labels[i]]]))

    # Get training time and append it, then append the MAPE below
    end_time = time.perf_counter()
    training_time = end_time-start_time
    time_list = ["Tiempo: ", training_time]
    results.append(time_list)
    results = results + buffer_MAPE

    return results


#ACTUALIZAR IGUA QUE LAYER 2
def train_cont_model_3layer(features, labels, param):
    # Name of the model to train
    print("\nContinuous training of a three layer neural network: First layer = " + str(param[0]) + " neurones, First dropout  = " + str(param[1]) + ", Second layer = " + str(param[2]) + " neurones, Second dropout  = " + str(param[3]) + ", Third layer = " + str(param[4]) + " neuronas")
    model_name = "FL = " + str(param[0]) + ", D1 = " + str(param[1]) + ", SL = " + str(param[2]) + ", D2 = " + str(param[3]) + ", TL = " + str(param[4])
    name_list = ["Model:", model_name]
    
    # Nested lists with results to return
    results = []
    results.append(name_list)
        
    start_time = time.perf_counter()

    # Create the 3 layer neural network
    model = keras.Sequential([
            layers.Input(shape=(features.shape[1],)),
            layers.Dense(param[0], activation="relu"),
            layers.Dropout(param[1]),
            layers.Dense(param[2], activation="relu"),
            layers.Dropout(param[3]),
            layers.Dense(param[4], activation="relu"),
            layers.Dense(1)
        ])
    model.compile(optimizer="adam", loss="mean_absolute_percentage_error")
    
    # Train the network and save the errors
    for i in range(labels.size):
        test_loss = float(abs(labels[i]-model.predict_on_batch(features.iloc[[i]]))/labels[i]*100)
        loss = [i, test_loss]
        results.append(loss)
        model.train_on_batch(x=features.iloc[[i]], y=np.array([[labels[i]]]))

    # Get training time and insert it in the nested list, below model name and before MAPEs
    end_time = time.perf_counter()
    training_time = end_time-start_time
    time_list = ["Tiempo: ", training_time]
    results.insert(time_list, 1)

    return results



def rolling_mean(data):
    # Extract the MAPE values from the nested list
    error_data = data[2:]
    rolling_mean = [point[1] for point in error_data]

    # Compute the rolling mean for 1000 values
    mean_buffer = rolling_mean[:1000]
    rolling_mean[999] = np.mean(mean_buffer)
    j = 0
    for i in range(1000, rolling_mean.size):
        mean_buffer[j] = rolling_mean[i]
        rolling_mean[i] = np.mean(mean_buffer)
        j += 1
        if j == 1000:
            j = 0
    return rolling_mean



def plot_best_models(model_type, Model_1, river_model = None,  Model_2 = None, Model_3 = None):
    # Matplotlib configuration
    mpl.rcParams["figure.figsize"] = (20, 12)
    # Remove top and right frame
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.bottom"] = True

    # Create a 2x2 grid of subplots within the same figure
    fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, constrained_layout=False)
    fig.supxlabel("Number of Observations")
    fig.suptitle("MAPE of " + model_type + " models")

    # Plot model metrics
    if river_model != None:
        ax1.plot(river_model,label="River model",color="tab:blue")
    ax1.plot(Model_1,label="Model_1",color="tab:green")
    if Model_2 != None:
        ax1.plot(Model_2,label="Model 2",color="tab:red")
    if Model_3 != None:
        ax1.plot(Model_3,label="Model 3",color="tab:brown")

    # Set Y limit
    if model_type == "TP":
        ax1.set_ylim([-0.5, 15.5])
    if model_type == "BP":
        ax1.set_ylim([-0.5, 15.5])
    if model_type == "Time":
        ax1.set_ylim([-0.5, 60.5])

    # Set Y label, grid and legend
    ax1.set_ylabel("MAPE error", color="k")
    ax1.tick_params(axis="y", labelcolor="k")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()  # Adjust subplot spacing

    # Plot the figure
    plt.savefig("Error_comparison_cont_model_" + model_type + ".tif", dpi = 500)
    plt.show()



def train_infer_times(features, labels, param):
    # Multilista en la que guardar resultados
    results = []

    #Definimos modelo de dos capas
    model = keras.Sequential([
            layers.Input(shape=(features.shape[1],)),
            layers.Dense(param[0], activation="relu"),
            layers.Dropout(param[1]),
            layers.Dense(param[2], activation="relu"),
            layers.Dense(1)
        ])
    model.compile(optimizer="adam", loss="mean_absolute_percentage_error")
    
    # Calculamos tiempo de inferencia solo
    start_time = time.perf_counter()
    for i in range(labels.size):
        test_loss = float(abs(labels[i]-model.predict_on_batch(features.iloc[[i]]))/labels[i]*100)
    end_time = time.perf_counter()
    infering_time = end_time-start_time

    # Calculamos tiempo de entrenamiento solo
    start_time = time.perf_counter()
    for i in range(labels.size):
        model.train_on_batch(x=features.iloc[[i]], y=np.array([[labels[i]]]))
    end_time = time.perf_counter()
    training_time = end_time-start_time
    
    total_time = training_time + infering_time
    prop_train_time = training_time/total_time*100
    prop_infer_time = infering_time/total_time*100

    model_name = "FL = " + str(param[0]) + ", D = " + str(param[1]) + ", SL = " + str(param[2])
    name_list = ["Model:", model_name, ""]
    results.append(name_list)
    time_list1 = ["Total time: ", total_time, ""]
    results.append(time_list1)
    time_list2 = ["Infer time: ", infering_time, prop_infer_time]
    results.append(time_list2)
    time_list3 = ["Train time: ", training_time, prop_train_time]
    results.append(time_list3)

    return results