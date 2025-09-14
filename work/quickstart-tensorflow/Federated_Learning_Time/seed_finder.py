import keras
import pandas as pd
from keras import layers, initializers
import os
import numpy as np
import random

dataset_PATH = "/home/eduardof/work/quickstart-tensorflow"
os.chdir(dataset_PATH)
dataset = pd.read_pickle("test_dataset.pkl")
# Extract features
features = dataset.drop(["Top power", "Bottom power", "Time"], axis=1)
# Extract labels time labels
Time = dataset.iloc[:,15]

error = 0
while error < 300:
    #seed = random.randint(0, 2**32 - 1)
    seed = 1
    model = keras.Sequential(
        [
            layers.Input(shape=(15,)),
            layers.Dense(35, activation="relu", kernel_initializer=initializers.RandomNormal(seed=seed) , bias_initializer=initializers.zeros()),
            layers.Dense(30, activation="relu", kernel_initializer=initializers.RandomNormal(seed=seed) , bias_initializer=initializers.zeros()),
            #layers.Dropout(0), commented due to dropout being 0 having the same effect as not having the layer in the first place
            layers.Dense(15, activation="relu", kernel_initializer=initializers.RandomNormal(seed=seed) , bias_initializer=initializers.zeros()),
            layers.Dense(1, kernel_initializer=initializers.RandomNormal(seed=seed) , bias_initializer=initializers.zeros()),
        ]
    )

    optimizer = keras.optimizers.Adam(0.001)
    model.compile(
        optimizer=optimizer,
        loss="mean_absolute_percentage_error",
        metrics=["mean_absolute_percentage_error"],
    )
    buffer_MAPE = []
    for i in range(Time.size):
        test_loss = abs(float(Time.iloc[i])-float(model.predict_on_batch(features.iloc[[i]])))/float(Time.iloc[i])*100

        loss_dummy = [test_loss]
        buffer_MAPE.append(loss_dummy)

    error = float(np.mean(buffer_MAPE))
    print(f"Error of seed {seed} = {error}")

print(f"Seed_2 = {seed}")