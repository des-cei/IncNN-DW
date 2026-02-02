"""tfexample: A Flower / TensorFlow app."""
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from Program.task import load_data_clients, load_model
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
        bs,
    ):
        self.model = load_model(learning_rate)
        self.x_train, self.y_train, self.x_val, self.y_val = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.partition_id = partition_id
        self.batch_size = bs

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        self.model.set_weights(parameters)

        if self.batch_size == 1:
            features = self.x_train
            label = self.y_train
            for i in range(self.y_train.size):
                self.model.train_on_batch(x=features.iloc[[i]], y=np.array([[label.iloc[i]]]))

        else:
            for i in range(self.y_train.size//self.batch_size):
                features_batch = self.x_train.iloc[(i*self.batch_size):(i*self.batch_size+self.batch_size)]
                labels_batch = self.y_train.iloc[(i*self.batch_size):(i*self.batch_size+self.batch_size)]
                self.model.train_on_batch(x=features_batch, y=np.array([[labels_batch]]))

            features_batch = self.x_train.iloc[(i*self.batch_size):]
            labels_batch = self.y_train.iloc[(i*self.batch_size):]
            self.model.train_on_batch(x=features_batch, y=np.array([[labels_batch]]))

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.model.set_weights(parameters)
        features = self.x_val
        label = self.y_val
        buffer_MAPE = []
        for i in range(label.size):
            test_loss = abs(float(label.iloc[i])-float(self.model.predict_on_batch(features.iloc[[i]])))/float(label.iloc[i])*100

            loss_dummy = [test_loss]
            buffer_MAPE.append(loss_dummy)

        loss = np.mean(buffer_MAPE)

        return float(loss), len(self.y_val), {"mean_absolute_percentage_error": float(loss)}

def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    data = load_data_clients(partition_id)

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    learning_rate = context.run_config["learning-rate"]

    # Define Batch Size
    bs = 1

    # Return Client instance
    return FlowerClient(learning_rate, data, epochs, batch_size, verbose, partition_id, bs).to_client()

# Flower ClientApp
app = ClientApp(client_fn=client_fn)