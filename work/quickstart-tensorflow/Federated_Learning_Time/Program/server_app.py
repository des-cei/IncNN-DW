"""tfexample: A Flower / TensorFlow app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters, NDArrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, Bulyan, FaultTolerantFedAvg, FedAdagrad, FedAdam, FedAvgAndroid, FedAvgM, FedMedian, FedOpt, FedProx, FedTrimmedAvg
from flwr.server.strategy import FedYogi, Krum, QFedAvg
from Federated_Learning_Time.Program.task import load_model

import os
import pandas as pd
import pickle
from flwr.common.logger import log
from logging import INFO
import numpy as np


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    mape = [num_examples * m["mean_absolute_percentage_error"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"mean_absolute_percentage_error": sum(mape) / sum(examples)}

# The `evaluate` function will be called by Flower after every round
def evaluate(server_round: int, parameters, config):
    dataset_PATH = "/home/eduardof/work/quickstart-tensorflow"
    os.chdir(dataset_PATH)
    dataset = pd.read_pickle("test_dataset.pkl")
    # Extract features
    features = dataset.drop(["Top power", "Bottom power", "Time"], axis=1)
    # Extract labels time labels
    Time = dataset.iloc[:,15]

    model = load_model()
    model.set_weights(parameters)
    buffer_MAPE = []
    for i in range(Time.size):
        test_loss = abs(float(Time.iloc[i])-float(model.predict_on_batch(features.iloc[[i]])))/float(Time.iloc[i])*100

        loss_dummy = [test_loss]
        buffer_MAPE.append(loss_dummy)

    loss = np.mean(buffer_MAPE)
    print(f"Server-side round {server_round} evaluation loss {loss}")

    store_path = "/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results/parameters_models/server_round_" + str(server_round)
    os.chdir(store_path)
    with open("parameters_model_server.pkl", 'wb') as h:
        pickle.dump(parameters, h, protocol=pickle.HIGHEST_PROTOCOL)




def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""


    # Let's define the global model and pass it to the strategy
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define the strategy
    strategy = strategy = Krum(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=1.0,
        min_available_clients=3,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        #save_path="/home/eduardof/work/quickstart-tensorflow/Federated_Learning_Time/Results",
        #proximal_mu = 0.001, #Term needed for fed prox --> Values: 1, 0.1, 0.01, 0.001
        evaluate_fn= evaluate
    )
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
