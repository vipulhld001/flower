import torch

# FedAvg Algorithm
def fed_avg(global_model, client_models):
    print("[INFO] Performing FedAvg...")
    # Initialize global model parameters as zeros
    global_state_dict = global_model.state_dict()
    for key in global_state_dict:
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])

    # Average the parameters from each client
    for client_model in client_models:
        client_state_dict = client_model.state_dict()
        for key in global_state_dict:
            global_state_dict[key] += client_state_dict[key] / len(client_models)

    # Update the global model with averaged parameters
    global_model.load_state_dict(global_state_dict)
    return global_model

# FedProx Algorithm
def fed_prox(global_model, client_models, mu=0.1):
    print("[INFO] Performing FedProx...")
    global_state_dict = global_model.state_dict()

    # Average parameters with FedProx regularization
    for key in global_state_dict:
        aggregated_param = torch.zeros_like(global_state_dict[key])
        for client_model in client_models:
            client_state_dict = client_model.state_dict()
            aggregated_param += client_state_dict[key] / len(client_models)
        global_state_dict[key] = aggregated_param

    # Apply proximal term to regularize global model
    for key in global_state_dict:
        global_state_dict[key] = global_state_dict[key] - mu * (global_state_dict[key] - global_model.state_dict()[key])

    # Update global model
    global_model.load_state_dict(global_state_dict)
    return global_model

# FedOpt Algorithm (Adaptive optimization using Adam)
def fed_opt(global_model, client_models, optimizer, lr=0.01):
    print("[INFO] Performing FedOpt...")
    global_state_dict = global_model.state_dict()

    # Initialize optimizer
    opt = optimizer(global_model.parameters(), lr=lr)

    # Average the parameters from each client
    for key in global_state_dict:
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])
        for client_model in client_models:
            client_state_dict = client_model.state_dict()
            global_state_dict[key] += client_state_dict[key] / len(client_models)

    # Update global model parameters using optimizer
    opt.zero_grad()
    for param_name, param in global_model.named_parameters():
        param.grad = global_state_dict[param_name] - param.data
    opt.step()

    return global_model

# Algorithm Selector
def get_federated_algo(algo_name, **kwargs):
    print(f"[INFO] Initializing federated algorithm: {algo_name}")
    if algo_name == "FedAvg":
        return fed_avg
    elif algo_name == "FedProx":
        return lambda global_model, client_models: fed_prox(global_model, client_models, mu=kwargs.get('mu', 0.1))
    elif algo_name == "FedOpt":
        return lambda global_model, client_models: fed_opt(global_model, client_models, optimizer=kwargs.get('optimizer', torch.optim.Adam), lr=kwargs.get('lr', 0.01))
    else:
        raise ValueError(f"[ERROR] Unknown federated algorithm: {algo_name}")
