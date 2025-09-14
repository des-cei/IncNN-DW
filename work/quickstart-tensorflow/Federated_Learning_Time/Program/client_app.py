"""tfexample: A Flower / TensorFlow app."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from Federated_Learning_Time.Program.task import load_data, load_model

import os
from flwr.common.logger import log
from logging import INFO, WARN
import pickle
import numpy as np

# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        learning_rate,
        data,
        epochs,
        batch_size,
        verbose,
        partition_id,
    ):
        self.model = load_model(learning_rate)
        self.x_train, self.y_train, self.x_val, self.y_val = data #, self.x_test, self.y_test = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.partition_id = partition_id

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        self.model.set_weights(parameters)
        #self.model.fit(
        #    self.x_train,
        #    self.y_train,
        #    epochs=self.epochs,
        #    batch_size=self.batch_size,
        #    verbose=self.verbose,
        #)

        ### batch size == 1
        """
        features = self.x_train
        label = self.y_train
        for i in range(self.y_train.size):
            self.model.train_on_batch(x=features.iloc[[i]], y=np.array([[label.iloc[i]]]))
        """

        ### batch size != 1
        bs = 50
        for i in range(self.y_train.size//bs):
            features_batch = self.x_train.iloc[(i*bs):(i*bs+bs)]
            labels_batch = self.y_train.iloc[(i*bs):(i*bs+bs)]
            self.model.train_on_batch(x=features_batch, y=np.array([[labels_batch]]))

        features_batch = self.x_train.iloc[(i*bs):]
        labels_batch = self.y_train.iloc[(i*bs):]
        self.model.train_on_batch(x=features_batch, y=np.array([[labels_batch]]))

        """
        features = self.x_val
        label = self.y_val
        buffer_MAPE = []
        for i in range(label.size):
            test_loss = abs(float(label.iloc[i])-float(self.model.predict_on_batch(features.iloc[[i]])))/float(label.iloc[i])*100

            loss_dummy = [test_loss]
            buffer_MAPE.append(loss_dummy)

        loss = np.mean(buffer_MAPE)

        #log(WARN, f"Client-side {self.partition_id} evaluation loss {loss}")
        print(f"Client-side {self.partition_id} evaluation loss {loss}")

        del buffer_MAPE

        features = self.x_test
        label = self.y_test
        buffer_MAPE = []
        for i in range(label.size):
            test_loss = abs(float(label.iloc[i])-float(self.model.predict_on_batch(features.iloc[[i]])))/float(label.iloc[i])*100

            loss_dummy = [test_loss]
            buffer_MAPE.append(loss_dummy)

        loss = np.mean(buffer_MAPE)

        #log(WARN, f"Client-side {self.partition_id} test loss {loss}")
        print(f"Client-side {self.partition_id} test loss {loss}")
        """

        #store_path = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/parameters_models/server_round_" + str(self.group_id)
        #os.chdir(store_path)
        #with open("parameters_model_client_" + str(self.partition_id) + ".pkl", 'wb') as h:
        #    pickle.dump(parameters, h, protocol=pickle.HIGHEST_PROTOCOL)
        #log(INFO, f"Client checkpoint saved")

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        #Evaluate the model on the data this client has.
        self.model.set_weights(parameters)
        #loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        features = self.x_val
        label = self.y_val
        buffer_MAPE = []
        for i in range(label.size):
            test_loss = abs(float(label.iloc[i])-float(self.model.predict_on_batch(features.iloc[[i]])))/float(label.iloc[i])*100

            loss_dummy = [test_loss]
            buffer_MAPE.append(loss_dummy)

        loss = np.mean(buffer_MAPE)

        #log(WARN, f"Server-side evaluation loss {loss}")
        """
        print(f"Server-side evaluation loss {loss}")

        del buffer_MAPE

        features = self.x_test
        label = self.y_test
        buffer_MAPE = []
        for i in range(label.size):
            test_loss = abs(float(label.iloc[i])-float(self.model.predict_on_batch(features.iloc[[i]])))/float(label.iloc[i])*100

            loss_dummy = [test_loss]
            buffer_MAPE.append(loss_dummy)

        loss = np.mean(buffer_MAPE)

        #log(WARN, f"Server-side test loss {loss}")
        print(f"Server-side test loss {loss}")
        """
        #if self.partition_id == 0:
        #    store_path = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/parameters_models/server_round_" + str(self.server_round)
        #    os.chdir(store_path)
        #    with open("parameters_model_server.pkl", 'wb') as h:
        #        pickle.dump(parameters, h, protocol=pickle.HIGHEST_PROTOCOL)
        #    log(INFO, f"Client checkpoint saved")

        return float(loss), len(self.y_val), {"mean_absolute_percentage_error": float(loss)}



def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    """
    with open("context.pkl", "wb") as NN_2layers_dict:
        pickle.dump(context, NN_2layers_dict)
    NN_2layers_dict.close()
    """
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(learning_rate, data, epochs, batch_size, verbose, partition_id).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
