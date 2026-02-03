This repository is divided in two parts, Model_Architecture_Research studies different model architectures (DNN, LSTM) trained incrementally. A grid search algorithm is employed and pareto graphs are plotted to optimize the hyperparameters of each model type. After the best model architectures are selected, these are used on the second part, FL_Implementation, where 12 aggregations are studied by training 3 clients on independent datasets and aggregating them on the server model. This is done for one round, after which pareto graphs are ploted to compare the strategies performance.

Required libraries (used versions are indicated in case future versions lead to incompatibilities):
flwr 1.18.0
keras 3.9.2
matplotlib 3.10.3
mplcursors 0.6
numpy 2.1.3
openpyxl 3.1.5
pandas 2.2.3
river 0.22.0
scikit-learn 1.6.1
scipy 1.15.2
tensorflow 2.19.0



###########################
Model_Architecture_Research
###########################

To execute this programs, the current working directory must be the Model_Architecture_Research folder. Inside this folder there are 4 folders for collecting results and one folder for the python codes. There are three types of codes in this section, Modeling, Dictionary_management and Pareto:

The modeling files consist on:
- LSTM_multiprocessing.py --> Performs a grid-search algorithm to optimize the hyperparameters of a base LSTM model.
- inc_modeling_HART_baseline.py --> Trains and tests incrementally the baseline architecture.
- Best_models_train.py --> Trains the selected best models for this work independently.
- 3LNN_multiprocessing.py --> Performs a grid-search algorithm to optimize the hyperparameters of a 3 layered DNN.
- 2LNN_multiprocessing.py --> Perfirms a grid-search algorithm to optimize the hyperparameters of a 2 layered DNN.
It should be noticed this programs are computationally expensive. Due to this, the grid-search is separated in different processes to run in parallel to reduce execution time.

The Dictionary_management programs work in pairs, where:
- dictionary_combine_{model_architecture}.py --> When results are obtained in groups (for example due to memory contraints) this program combines the dictionaries into one.
- dictionary_remake_size_{model_architecture}.py --> When the user wants to plot pareto graph, this program resizes and processes the dictionary to be be later used to generate paretos.

The pareto programs prepare pareto graphs of the performance indicators obtained during the grid-search to then compare and decide on the best hyperparameter combination.



#################
FL_Implementation
#################

To execute this programs, the current working directory must be the FL_Implementation folder. Furthermore, it is necessary to modify two functions inside the flwr library. In Flwr/server/server.py:
- The user must import os, pickle and time libraries at the start file
- Function fit_round() should be substituted by:
def fit_round(
    self,
    server_round: int,
    timeout: Optional[float],
) -> Optional[
    tuple[Optional[Parameters], dict[str, Scalar], FitResultsAndFailures]
]:
    """Perform a single round of federated averaging."""
    # Get clients and their respective instructions from strategy
    client_instructions = self.strategy.configure_fit(
        server_round=server_round,
        parameters=self.parameters,
        client_manager=self._client_manager,
    )

    if not client_instructions:
        log(INFO, "configure_fit: no clients selected, cancel")
        return None
    log(
        INFO,
        "configure_fit: strategy sampled %s clients (out of %s)",
        len(client_instructions),
        self._client_manager.num_available(),
    )

    # Collect `fit` results from all clients participating in this round
    results, failures = fit_clients(
        client_instructions=client_instructions,
        max_workers=self.max_workers,
        timeout=timeout,
        group_id=server_round,
    )
    log(
        INFO,
        "aggregate_fit: received %s results and %s failures",
        len(results),
        len(failures),
    )

    start_time = time.perf_counter()
    # Aggregate training results
    aggregated_result: tuple[
        Optional[Parameters],
        dict[str, Scalar],
    ] = self.strategy.aggregate_fit(server_round, results, failures)
    end_time = time.perf_counter()

    aggregate_time = end_time - start_time

    dataset_path_path = os.path.dirname(os.getcwd())
    path = os.path.join(
        dataset_path_path,
        "Results",
        "provisionary_parameters_models",
        f"server_round_{server_round}",
        "aggregate_time.pkl"
    )
    with open(path, "wb") as h:
        pickle.dump(aggregate_time, h, protocol=pickle.HIGHEST_PROTOCOL)

    parameters_aggregated, metrics_aggregated = aggregated_result
    return parameters_aggregated, metrics_aggregated, (results, failures)



- Function fit_client() should be substituted by:
def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float], group_id: int, index: int
) -> tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout, group_id=group_id)
    params = fit_res.parameters

    dataset_path_path = os.path.dirname(os.getcwd())
    path = os.path.join(
        dataset_path_path,
        "Results",
        "provisionary_parameters_models",
        f"server_round_{group_id}",
        f"parameters_model_client_{index+1}.pkl"
    )
    with open(path, "wb") as h:
        pickle.dump(params, h, protocol=pickle.HIGHEST_PROTOCOL)

    return client, fit_res



Inside the FL_Implementation folder, the user can find the dataset used, a program to generate three independent subdatasets, and an auxiliary functions program for the framework. Then four folders are found, one for results and one to implement FL techniques with each type of model (Top Power, Bottom Power and Time). Each of these last three folders share the same structure:

Federated_Learning_{Model_type}
- A folder named Program containing three files: server_app.py in charge of describing the server behaviour in a FL implementation using Flwr library, client_app.py in charge of describing the client behaviour, and task.py which defines how each client/server loads their model and respective dataset.
- A pyproject.toml file that defines some configuration parameters for a Flwr implementation
- A multi_flwr_{Model_type}.py program. This program performs a 10-fold cross-validation of 12 strategies by executing subprocesses, running single FL implementation using the files described above. It should be noted in order to perform both the cross-validation and changing the strategy, the files on the Program folder are parsed and the specific line containing the desired information is modified. Due to this, if any change is done to those programs, the variable signaling the line number should also be modified. This file saves the results on the result folder.

Results Folder
- Contains 5 folders to save resuls:
    - Testing dictionaries --> stores the complete dictionary of the 10 fold cross-validation of the 12 strategies
    - provisionary_parameters_models --> saves the model parametrs of the server and clients on every round to during FL implementation to be evaluated after implementation is finished
    - Pareto_{} --> Stores the pareto images obtained
- Contains a dictionary_combine_FL.py file that combines the dictionary of each model type into one
- Contains three dictionar_remake_size_FL.py files to prepare the dictionary for different paretos:
    - dictionar_remake_size_FL.py --> Prepare the dictionary to plot paretos per model type of all server results
    - dictionar_remake_size_FL_server_no_outliers.py --> Prepare the dictionary to plot  per model type of server results excluding those considered outliers to better appreciate the frontier
    - dictionar_remake_size_FL_clients&server_no_outliers.py --> Prepare the dictionary to plot  per model type of clients and server results, (excluding those considered outliers to better appreciate the frontier) to compare performance of srver and clients
- Three pareto_FL.py files that plot the three types of paretos mentioned above
