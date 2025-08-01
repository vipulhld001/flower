"""awesome: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from awesome.task import Net, get_weights, set_weights, test, get_transforms

from typing import List, Tuple
from datasets import load_dataset
from torch.utils.data import DataLoader
import json

def get_evaluate_fn(testloader, device):
    
    def evaluate_cent(server_round, parameters_ndarrays, config):
        
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        
        return loss, {"cent_accuracy": accuracy}
    
    return evaluate_cent
        
        

#For Accuracy
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """A function that aggregates metrics """
    accuracies = [num_examples *m ['accuracy'] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    return {"accuracy" : sum(accuracies)/ total_examples}

###to print from client app (1)
def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:

    b_values = []
    for _, m in metrics:
        my_metrics = m.get("my_metrics", None)
        print(my_metrics)
        my_metrics_dict = json.loads(my_metrics)
        b_values.append(my_metrics_dict.get("b", 0))

    
    
    return {"max_b": max(b_values)}


#For Config and passing my own parameters

def on_fit_config(server_round:int ) -> Metrics:
    """Add new h parameter"""
    lr = 0.01
    if server_round > 2:
        lr = 0.0005
    return {"lr": lr}







def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)
    
    ###Create testloader
    testset = load_dataset("zalando-datasets/fashion_mnist")["test"]
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size = 32)
    

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn = weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn = on_fit_config,
        evaluate_fn = get_evaluate_fn(testloader, device="cpu") ,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
