import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import openpyxl
import random

dataset = pd.read_pickle("dataset_complete.pkl")

dataset_client_1 = []
dataset_client_2 = []
dataset_client_3 = []
for i in range(dataset.shape[0]):
    dummy_df = dataset.iloc[[i]]
    dummy_list = [float(dummy_df.values[0,0]), float(dummy_df.values[0,1]), float(dummy_df.values[0,2]),
                  int(dummy_df.values[0,3]), int(dummy_df.values[0,4]), int(dummy_df.values[0,5]),
                  int(dummy_df.values[0,6]), int(dummy_df.values[0,7]), int(dummy_df.values[0,8]),
                  int(dummy_df.values[0,9]), int(dummy_df.values[0,10]), int(dummy_df.values[0,11]),
                  int(dummy_df.values[0,12]), int(dummy_df.values[0,13]), int(dummy_df.values[0,14]),
                  float(dummy_df.values[0,15]), float(dummy_df.values[0,16]), float(dummy_df.values[0,17])]
    if ((dummy_df.iat[0,8] == 0) and (dummy_df.iat[0,9] == 0) and (dummy_df.iat[0,10] == 0) and (dummy_df.iat[0,11] == 0) and (dummy_df.iat[0,12] == 0) and (dummy_df.iat[0,13] == 0) and (dummy_df.iat[0,14] == 0)):
        dataset_client_1.append(dummy_list)
    elif ((dummy_df.iat[0,4] == 0) and (dummy_df.iat[0,5] == 0) and (dummy_df.iat[0,6] == 0) and (dummy_df.iat[0,7] == 0) and (dummy_df.iat[0,12] == 0) and (dummy_df.iat[0,13] == 0) and (dummy_df.iat[0,14] == 0)):
        dataset_client_2.append(dummy_list)
    elif ((dummy_df.iat[0,4] == 0) and (dummy_df.iat[0,5] == 0) and (dummy_df.iat[0,6] == 0) and (dummy_df.iat[0,7] == 0) and (dummy_df.iat[0,8] == 0) and (dummy_df.iat[0,9] == 0) and (dummy_df.iat[0,10] == 0) and (dummy_df.iat[0,11] == 0)):
            dataset_client_3.append(dummy_list)

random.shuffle(dataset_client_1)
random.shuffle(dataset_client_2)
random.shuffle(dataset_client_3)

final_dataset_client_1 = []
final_dataset_client_2 = []
final_dataset_client_3 = []
test_dataset = []

header = ['user', 'kernel', 'idle', 'Main', 'aes', 'bulk', 'crs', 'kmp', 'knn', 'merge', 'nw', 'queue', 'stencil2d', 'stencil3d', 'strided', 'Top power', 'Bottom power', 'Time']
final_dataset_client_1.append(header)
final_dataset_client_2.append(header)
final_dataset_client_3.append(header)
test_dataset.append(header)

final_dataset_client_1 = final_dataset_client_1 + dataset_client_1
final_dataset_client_2 = final_dataset_client_2 + dataset_client_2
final_dataset_client_3 = final_dataset_client_3 + dataset_client_3
test_dataset = test_dataset + dataset_client_1[-4020:] + dataset_client_2[-4315:] + dataset_client_3[-3796:]

df_final_dataset_client_1 = pd.DataFrame(final_dataset_client_1[1:],columns=final_dataset_client_1[0])
df_final_dataset_client_2 = pd.DataFrame(final_dataset_client_2[1:],columns=final_dataset_client_2[0])
df_final_dataset_client_3 = pd.DataFrame(final_dataset_client_3[1:],columns=final_dataset_client_3[0])
df_test_dataset = pd.DataFrame(test_dataset[1:],columns=test_dataset[0])


df_final_dataset_client_1.to_pickle('dataset_client_1.pkl')
df_final_dataset_client_2.to_pickle('dataset_client_2.pkl')
df_final_dataset_client_3.to_pickle('dataset_client_3.pkl')