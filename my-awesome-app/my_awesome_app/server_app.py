"""my-awesome-app: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, Metrics
from typing import List, Tuple
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from my_awesome_app.task import Net, get_weights, set_weights, test, get_transformss
from datasets import load_dataset

from torch.utils.data import DataLoader



#Evaluation the global model (Centralised model) on the test data

def get_evaluate_fn(testloader, device):
    """Return an evaluation function for the provided test dataset."""
    def evaluate(server_round, parameters_ndarrays, config):

        net = Net()
        set_weights(net, parameters_ndarrays) #set the value
        net.to(device)                  #send it to the device
        loss, accuracy = test(net, testloader, device) #test the model

        return loss, {"centralised_accuracy": accuracy} #return the loss and accuracy
    return evaluate




def weighted_average(metrics: List[Tuple[int, Metrics]])-> Metrics:
    """A method to calculate the weighted average of metrics"""

    accuracies = [num_examples * metric["accuracy"] for num_examples, metric in metrics]
    #only_accuracies = [metric["accuracy"] for num_examples, metric in metrics]
    total_examples = sum([num_examples for num_examples, metric in metrics])

    return {
        "accuracy": sum(accuracies) / total_examples
    } #accuracy comes from here

def on_fit_config(server_round: int) -> Metrics:
    """Return a configuration with a learning rate for the client."""
    lr = 0.01
    if server_round >2:
        lr = 0.001
    return {"lr": lr}




def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"] #No of server rounds
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    #testloader for global test
    testset = load_dataset("zalando-datasets/fashion_mnist")["test"]
    
    testloader = DataLoader(testset.with_transform(get_transformss()), batch_size=32)



    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn= weighted_average,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device="cuda:0"),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
