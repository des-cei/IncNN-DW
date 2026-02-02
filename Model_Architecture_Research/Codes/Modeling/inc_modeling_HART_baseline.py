import os
import tensorflow as tf
import numpy as np
import pandas as pd
import river
from river import metrics, preprocessing, forest
import time
import pickle


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dataset = pd.read_pickle("dataset.pkl")

# Path and dictionary to store obtained results
os.chdir("Results_Baseline")
model_types = ['Top-power', 'Bottom-power', 'Time']
resultados = {'Top-power': {}, 'Bottom-power': {}, 'Time': {}}
for mt in model_types :
    resultados[mt] = {'MAPE-mean-error': {}, 'MAPE-sdv-error': {}, 'MAPE-mean-error_cv': {}, 'Train-time': {}, 'Infer-time': {}}

# Extract features
features_df = dataset.drop(["Top power", "Bottom power", "Time"], axis=1)

# Extract labels
labels_df = dataset[["Top power", "Bottom power", "Time"]]

# PS power model
top_power_model = (
                river.preprocessing.StandardScaler() |
                river.tree.HoeffdingAdaptiveTreeRegressor(
                    max_depth=100,
                    grace_period=50,
                    model_selector_decay=0.05,
                    seed=42
                )
            )

# PL power models
bottom_power_model = (
                river.preprocessing.StandardScaler() |
                river.tree.HoeffdingAdaptiveTreeRegressor(
                    max_depth=100,
                    grace_period=50,
                    model_selector_decay=0.05,
                    seed=42
                )
            )

# Execution time model
time_model = tmp_model = river.forest.ARFRegressor(seed=42, max_features=None, grace_period=50, n_models = 5, max_depth=100, model_selector_decay=0.05)

# Model MAPE metrics
top_power_mape = river.utils.Rolling(river.metrics.MAPE(), window_size=1000)
bottom_power_mape = river.utils.Rolling(river.metrics.MAPE(), window_size=1000)
time_mape = river.utils.Rolling(river.metrics.MAPE(), window_size=1000)

# List of metrics and models
models = [top_power_model, bottom_power_model, time_model]
metrics = [top_power_mape, bottom_power_mape, time_mape]

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



# Define results and auxiliary variables
cv = 5

MAPE_mean_TP = []
MAPE_std_TP = []
MAPE_mean_BP = []
MAPE_std_BP = []
MAPE_mean_Time = []
MAPE_std_Time = []

infer_time_TP = 0
train_time_TP = 0
infer_time_BP = 0
train_time_BP = 0
infer_time_Time = 0
train_time_Time = 0


# Loop over the observations
for j in range(cv):
    i = 0
    buffer_TP = []
    buffer_BP = []
    buffer_Time = []

    for features, labels in river.stream.iter_pandas(features_df, labels_df, shuffle=False, seed=42):
        # Features and labels accommodation
        features, labels = features_labels_accommodation(features, labels)

        for metric, label in zip(metrics, labels):
            start_time = time.perf_counter()
            # Make a prediction
            y_pred = top_power_model.predict_one(features)
            end_time = time.perf_counter()
            infer_time_TP = infer_time_TP + end_time - start_time

            start_time = time.perf_counter()
            # Train the model
            top_power_model.learn_one(features, label)
            # Update metric
            metric.update(label, y_pred)
            end_time = time.perf_counter()
            train_time_TP = train_time_TP + end_time - start_time

        for metric, label in zip(metrics, labels):
            start_time = time.perf_counter()
            # Make a prediction
            y_pred = bottom_power_model.predict_one(features)
            end_time = time.perf_counter()
            infer_time_BP = infer_time_BP + end_time - start_time

            start_time = time.perf_counter()
            # Train the model
            bottom_power_model.learn_one(features, label)
            # Update metric
            metric.update(label, y_pred)
            end_time = time.perf_counter()
            train_time_BP = train_time_BP + end_time - start_time

        for metric, label in zip(metrics, labels):
            start_time = time.perf_counter()
            # Make a prediction
            y_pred = time_model.predict_one(features)
            end_time = time.perf_counter()
            infer_time_Time = infer_time_Time + end_time - start_time

            start_time = time.perf_counter()
            # Train the model
            time_model.learn_one(features, label)
            # Update metric
            metric.update(label, y_pred)
            end_time = time.perf_counter()
            train_time_Time = train_time_Time + end_time - start_time

        if i >= 1000:
            buffer_TP.append(metrics[0])
            buffer_BP.append(metrics[1])
            buffer_Time.append(metrics[2])
        i += 1

    dummy_mean = np.mean(buffer_TP)
    MAPE_mean_TP.append(dummy_mean)
    dummy_std = np.mean(buffer_TP)
    MAPE_std_TP.append(dummy_std)

    dummy_mean = np.mean(buffer_BP)
    MAPE_mean_BP.append(dummy_mean)
    dummy_std = np.mean(buffer_BP)
    MAPE_std_BP.append(dummy_std)

    dummy_mean = np.mean(buffer_Time)
    MAPE_mean_Time.append(dummy_mean)
    dummy_std = np.mean(buffer_Time)
    MAPE_std_Time.append(dummy_std)

    del dummy_mean, dummy_std, buffer_TP, buffer_BP, buffer_Time

# Save results
resultados["Top-power"]['MAPE-mean-error'] = float(np.mean(MAPE_mean_TP))
resultados["Top-power"]['MAPE-sdv-error'] = float(np.mean(MAPE_std_TP))
resultados["Top-power"]['MAPE-mean-error_cv'] = float(np.std(MAPE_mean_TP))
resultados["Top-power"]['Train-time'] = float(train_time_TP/(cv*labels.size))
resultados["Top-power"]['Infer-time'] = float(infer_time_TP/(cv*labels.size))

resultados["Bottom-power"]['MAPE-mean-error'] = float(np.mean(MAPE_mean_BP))
resultados["Bottom-power"]['MAPE-sdv-error'] = float(np.mean(MAPE_std_BP))
resultados["Bottom-power"]['MAPE-mean-error_cv'] = float(np.std(MAPE_mean_BP))
resultados["Bottom-power"]['Train-time'] = float(train_time_BP/(cv*labels.size))
resultados["Bottom-power"]['Infer-time'] = float(infer_time_BP/(cv*labels.size))

resultados["Time"]['MAPE-mean-error'] = float(np.mean(MAPE_mean_Time))
resultados["Time"]['MAPE-sdv-error'] = float(np.mean(MAPE_std_Time))
resultados["Time"]['MAPE-mean-error_cv'] = float(np.std(MAPE_mean_Time))
resultados["Time"]['Train-time'] = float(train_time_Time/(cv*labels.size))
resultados["Time"]['Infer-time'] = float(infer_time_Time/(cv*labels.size))

with open("HART-baseline-dictionary.pkl", "wb") as NN_3layers_dict:
    pickle.dump(resultados, NN_3layers_dict)
NN_3layers_dict.close()