"""tfexample: A Flower / TensorFlow app."""
from typing import List, Tuple
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, Bulyan, FaultTolerantFedAvg, FedAdagrad, FedAdam, FedAvgM, FedMedian, FedOpt, FedProx, FedTrimmedAvg, FedYogi, Krum
from Program.task import load_model, load_data_server
import os
import pickle
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
    features, labels = load_data_server()

    model = load_model()
    model.set_weights(parameters)
    buffer_MAPE = []
    for i in range(labels.size):
        test_loss = abs(float(labels.iloc[i])-float(model.predict_on_batch(features.iloc[[i]])))/float(labels.iloc[i])*100

        loss_dummy = [test_loss]
        buffer_MAPE.append(loss_dummy)

    loss = np.mean(buffer_MAPE)
    print(f"Server-side round {server_round} evaluation loss {loss}")

    dataset_path_path = os.path.dirname(os.getcwd())
    path = os.path.join(
        dataset_path_path,
        "Results",
        "provisionary_parameters_models",
        f"server_round_{server_round}",
        "parameters_model_server.pkl"
    )
    with open(path, "wb") as h:
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
        #proximal_mu = 0.001, #Term needed for fed prox --> Values: 1, 0.1, 0.01, 0.001
        evaluate_fn= evaluate
    )
    # Read from config
    num_rounds = 1
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)