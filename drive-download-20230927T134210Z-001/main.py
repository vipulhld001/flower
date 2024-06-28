import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn, FlowerClient
from server import get_on_fit_config, get_evaluate_fn

################




#############################
    #NEW STRAT###


from typing import Callable, Union
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from model import Net, train, test


from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


class FedCustom(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        net = Net(10)
        flwrclinet  = FlowerClient()
        ndarrays = flwrclinet.get_parameters(config=net)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients







###################


# A decorator for Hydra. This tells hydra to by default load the config in conf/base.yaml
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    # Hydra automatically creates a directory for your experiments
    # by default it would be in <this directory>/outputs/<date>/<time>
    # you can retrieve the path to it as shown below. We'll use this path to
    # save the results of the simulation (see the last part of this main())
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare your dataset
    # When simulating FL workloads we have a lot of freedom on how the FL clients behave,
    # what data they have, how much data, etc. This is not possible in real FL settings.
    # In simulation you'd often encounter two types of dataset:
    #       * naturally partitioned, that come pre-partitioned by user id (e.g. FEMNIST,
    #         Shakespeare, SpeechCommands) and as a result these dataset have a fixed number
    #         of clients and a fixed amount/distribution of data for each client.
    #       * and others that are not partitioned in any way but are very popular in ML
    #         (e.g. MNIST, CIFAR-10/100). We can _synthetically_ partition these datasets
    #         into an arbitrary number of partitions and assign one to a different client.
    #         Synthetically partitioned dataset allow for simulating different data distribution
    #         scenarios to tests your ideas. The down side is that these might not reflect well
    #         the type of distributions encounter in the Wild.
    #
    # In this tutorial we are going to partition the MNIST dataset into 100 clients (the default
    # in our config -- but you can change this!) following a independent and identically distributed (IID)
    # sampling mechanism. This is arguably the simples way of partitioning data but it's a good fit
    # for this introductory tutorial.
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    ## 3. Define your clients
    # Unlike in standard FL (e.g. see the quickstart-pytorch or quickstart-tensorflow examples in the Flower repo),
    # in simulation we don't want to manually launch clients. We delegate that to the VirtualClientEngine.
    # What we need to provide to start_simulation() with is a function that can be called at any point in time to
    # create a client. This is what the line below exactly returns.
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. Define your strategy
    # A flower strategy orchestrates your FL pipeline. Although it is present in all stages of the FL process
    # each strategy often differs from others depending on how the model _aggregation_ is performed. This happens
    # in the strategy's `aggregate_fit()` method. In this tutorial we choose FedAvg, which simply takes the average
    # of the models received from the clients that participated in a FL round doing fit().
    # You can implement a custom strategy to have full control on all aspects including: how the clients are sampled,
    # how updated models from the clients are aggregated, how the model is evaluated on the server, etc
    # To control how many clients are sampled, strategies often use a combination of two parameters `fraction_{}` and `min_{}_clients`
    # where `{}` can be either `fit` or `evaluate`, depending on the FL stage. The final number of clients sampled is given by the formula
    # ``` # an equivalent bit of code is used by the strategies' num_fit_clients() and num_evaluate_clients() built-in methods.
    #         num_clients = int(num_available_clients * self.fraction_fit)
    #         clients_to_do_fit = max(num_clients, self.min_fit_clients)
    # ```
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.000,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=cfg.num_clients_per_round_fit,  # number of clients to sample for fit()
        fraction_evaluate=0.000,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # number of clients to sample for evaluate()
        min_available_clients=cfg.num_clients,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )  # a function to run on the server side to evaluate the global model.

    ## 5. Start Simulation
    # With the dataset partitioned, the client function and the strategy ready, we can now launch the simulation!
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # a function that spawns a particular client
        num_clients=cfg.num_clients,  # total number of clients
        config=fl.server.ServerConfig(
            num_rounds=cfg.num_rounds
        ),  # minimal config for the server loop telling the number of rounds in FL
        strategy=FedCustom(),
          # our strategy of choice
        client_resources={
            "num_cpus": 8,
            "num_gpus": 0.125,
        },  # (optional) controls the degree of parallelism of your simulation.
        # Lower resources per client allow for more clients to run concurrently
        # (but need to be set taking into account the compute/memory footprint of your workload)
        # `num_cpus` is an absolute number (integer) indicating the number of threads a client should be allocated
        # `num_gpus` is a ratio indicating the portion of gpu memory that a client needs. 8th of a GPU 0.125
    )

    # ^ Following the above comment about `client_resources`. if you set `num_gpus` to 0.5 and you have one GPU in your system,
    # then your simulation would run 2 clients concurrently. If in your round you have more than 2 clients, then clients will wait
    # until resources are available from them. This scheduling is done under-the-hood for you so you don't have to worry about it.
    # What is really important is that you set your `num_gpus` value correctly for the task your clients do. For example, if you are training
    # a large model, then you'll likely see `nvidia-smi` reporting a large memory usage of you clients. In those settings, you might need to
    # leave `num_gpus` as a high value (0.5 or even 1.0). For smaller models, like the one in this tutorial, your GPU would likely be capable
    # of running at least 2 or more (depending on your GPU model.)
    # Please note that GPU memory is only one dimension to consider when optimising your simulation. Other aspects such as compute footprint
    # and I/O to the filesystem or data preprocessing might affect your simulation  (and tweaking `num_gpus` would not translate into speedups)
    # Finally, please note that these gpu limits are not enforced, meaning that a client can still go beyond the limit initially assigned, if
    # this happens, your might get some out-of-memory (OOM) errors.

    ## 6. Save your results
    # (This is one way of saving results, others are of course valid :) )
    # Now that the simulation is completed, we could save the results into the directory
    # that Hydra created automatically at the beginning of the experiment.
    results_path = Path(save_path) / "results.pkl"

    # add the history returned by the strategy into a standard Python dictionary
    # you can add more content if you wish (note that in the directory created by
    # Hydra, you'll already have the config used as well as the log)
    results = {"history": history, "anythingelse": "here"}

    # save the results as a python pickle
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()


