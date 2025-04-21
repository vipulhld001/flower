import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocess_dataset import main as preprocess_dataset
from models import get_model
from federated_algos import get_federated_algo

# Hyperparameters and Configurations
BATCH_SIZE = 32
EPOCHS = 10
COMM_ROUNDS = 5
LEARNING_RATE = 0.0001
ALGO_NAME = "FedAvg"  # Options: "FedAvg", "FedProx", "FedOpt"
MODEL_NAME = "LSTM"  # Options: "MLP", "LSTM", "CNN", "Transformer"
FEATURES = ['dew point', 'humidity', 'precipitation', 'wind speed', 'air pressure']
TARGET = 'Temp'
NUM_CLIENTS = 4  # Number of clients

# Step 1: Dataset Class for PyTorch
class WeatherDataset(Dataset):
    # for mlp remove model name
    def __init__(self, data, features, target, model_name=None):
        self.data = data
        self.features = features
        self.target = target
        self.model_name = model_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx][self.features].values.astype(np.float32)
        y = self.data.iloc[idx][self.target].astype(np.float32)
        # Reshape for CNN
        if self.model_name == "CNN":
            x = x.reshape(len(self.features), 1)  # Correct: 5 channels, 1 sequence length
        elif self.model_name == "Transformer":
            x = x.reshape(len(self.features))  # For Transformer: [sequence_length, num_features]
        elif self.model_name == "LSTM":
            x = x.reshape(1, len(self.features))  # [sequence_length, input_dim]
        return torch.tensor(x), torch.tensor(y)

# Step 2: Train Local Model on Client Data
def train_local_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Calculate accuracy (within ±5% of true value)
            total_predictions += y_batch.size(0)
            correct_predictions += torch.sum(torch.abs(outputs.squeeze() - y_batch) <= 0.05 * y_batch).item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions * 100

        # Print loss and accuracy
        print(f"    [Client Training] Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Debugging predictions during the first epoch
        if epoch == 0:
            print(f"[DEBUG] Sample Predictions: {outputs[:5].detach().numpy().squeeze()}")
            print(f"[DEBUG] Target Values: {y_batch[:5].detach().numpy().squeeze()}")

# Step 3: Evaluate Model
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            total_loss += loss.item()
    return total_loss / len(data_loader)

# Step 4: Federated Learning Workflow (Updated)
# def federated_learning(clients_data, global_model, algo, criterion, epochs, comm_rounds):
#     # Dictionaries to store metrics
#     metrics = {
#         "communication_round": [],
#         "global_loss": [],
#         "client_id": [],
#         "local_epoch": [],
#         "local_loss": [],
#     }

#     global_loss = []
#     for round in range(comm_rounds):
#         print(f"[Federated Learning] Communication Round {round + 1}/{comm_rounds}")

#         # Train local models
#         client_models = []
#         local_losses = []  # Store local losses for this round
#         for client_id, data in enumerate(clients_data):
#             print(f"  [Client {client_id + 1}] Training local model...")
#             # local copy of the global model
#             local_model = get_model(MODEL_NAME, input_dim=len(FEATURES), output_dim=1)
#             #initialize local model with global model weights
#             # state dict - to predict order in which it is been used to forward weights
#             local_model.load_state_dict(global_model.state_dict())  # Start from global model
#             train_loader = DataLoader(WeatherDataset(data, FEATURES, TARGET), batch_size=BATCH_SIZE, shuffle=True)
#             optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

#             # Track local losses
#             for epoch in range(epochs):
#                 epoch_loss = 0
#                 local_model.train()
#                 for x_batch, y_batch in train_loader:
#                     optimizer.zero_grad()
#                     outputs = local_model(x_batch)
#                     loss = criterion(outputs, y_batch.unsqueeze(1))
#                     # backword propogate and update the losses
#                     loss.backward()
#                     optimizer.step()
#                     epoch_loss += loss.item()
                
#                 avg_epoch_loss = epoch_loss / len(train_loader)
#                 local_losses.append(avg_epoch_loss)

#                 # Save local loss for this epoch
#                 metrics["communication_round"].append(round + 1)
#                 metrics["client_id"].append(client_id + 1)
#                 metrics["local_epoch"].append(epoch + 1)
#                 metrics["local_loss"].append(avg_epoch_loss)

#                 print(f"    [Client {client_id + 1}] Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

#             client_models.append(local_model)

#         # Aggregate global model
#         global_model = algo(global_model, client_models)

#         # Evaluate global model
#         global_test_loss = evaluate_model(
#             global_model,
#             DataLoader(WeatherDataset(pd.concat(clients_data), FEATURES, TARGET), batch_size=BATCH_SIZE),
#             criterion,
#         )
#         global_loss.append(global_test_loss)

#         # Save global loss for this round
#         metrics["communication_round"].append(round + 1)
#         metrics["global_loss"].append(global_test_loss)
#         print(f"[Federated Learning] Round {round + 1} Global Loss: {global_test_loss:.4f}")

#     # Save metrics to Excel
#     save_metrics_to_excel(metrics)

#     return global_model, global_loss

def evaluate_global_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            total_loss += loss.item()

            # Calculate accuracy (within ±5% of true value)
            total_predictions += y_batch.size(0)
            correct_predictions += torch.sum(torch.abs(outputs.squeeze() - y_batch) <= 0.05 * y_batch).item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions * 100
    return avg_loss, accuracy
# Ensure all keys in metrics have the same length
# def pad_metrics(metrics):
#     max_len = max(len(v) for v in metrics.values())
#     for key, value in metrics.items():
#         while len(value) < max_len:
#             value.append(None)


def federated_learning(clients_data, global_model, algo, criterion, epochs, comm_rounds):
    # Dictionaries to store metrics
    metrics = {
        "communication_round": [],
        "global_loss": [],
        "global_accuracy": [],
        "client_id": [],
        "local_epoch": [],
        "local_loss": [],
        "local_accuracy": [],
    }

    global_loss = []
    global_accuracy = []

    for round in range(comm_rounds):
        print(f"[Federated Learning] Communication Round {round + 1}/{comm_rounds}")

        # Train local models
        client_models = []
        for client_id, data in enumerate(clients_data):
            print(f"  [Client {client_id + 1}] Training local model...")
            # Local copy of the global model
            local_model = get_model(MODEL_NAME, input_dim=len(FEATURES), output_dim=1)
            local_model.load_state_dict(global_model.state_dict())
            #train_loader = DataLoader(WeatherDataset(data, FEATURES, TARGET), batch_size=BATCH_SIZE, shuffle=True)
            # For CNN 
            train_loader = DataLoader(
                WeatherDataset(data, FEATURES, TARGET, MODEL_NAME),  # Pass MODEL_NAME
                batch_size=BATCH_SIZE,
                shuffle=True
            )

            optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

            # Track local losses and accuracy
            for epoch in range(epochs):
                epoch_loss = 0
                correct_predictions = 0
                total_predictions = 0

                local_model.train()
                for x_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = local_model(x_batch)
                    loss = criterion(outputs, y_batch.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                    # Calculate accuracy (within ±5% of true value)
                    total_predictions += y_batch.size(0)
                    correct_predictions += torch.sum(torch.abs(outputs.squeeze() - y_batch) <= 0.05 * y_batch).item()

                avg_epoch_loss = epoch_loss / len(train_loader)
                accuracy = correct_predictions / total_predictions * 100

                # Save metrics for this epoch
                metrics["communication_round"].append(round + 1)
                metrics["client_id"].append(client_id + 1)
                metrics["local_epoch"].append(epoch + 1)
                metrics["local_loss"].append(avg_epoch_loss)
                metrics["local_accuracy"].append(accuracy)

                print(f"    [Client {client_id + 1}] Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

            client_models.append(local_model)

        # Aggregate global model
        global_model = algo(global_model, client_models)

        # Evaluate global model
        global_test_loss, global_test_accuracy = evaluate_global_model(
            global_model,
            DataLoader(WeatherDataset(pd.concat(clients_data), FEATURES, TARGET), batch_size=BATCH_SIZE),
            criterion,
        )
        global_loss.append(global_test_loss)
        global_accuracy.append(global_test_accuracy)

        # Save global metrics
        metrics["communication_round"].append(round + 1)
        metrics["global_loss"].append(global_test_loss)
        metrics["global_accuracy"].append(global_test_accuracy)
        print(f"[Federated Learning] Round {round + 1} Global Loss: {global_test_loss:.4f}, Global Accuracy: {global_test_accuracy:.2f}%")

        # Ensure consistent lengths after each round
        pad_metrics(metrics)

    # Debugging: Check metric lengths before saving
    print("[DEBUG] Checking metrics lengths before saving...")
    for key, value in metrics.items():
        print(f"{key}: {len(value)}")

    # Save metrics to Excel
    save_metrics_to_excel(metrics)

    return global_model, global_loss, global_accuracy


# Padding function for metrics
def pad_metrics(metrics):
    """
    Ensures all metrics in the dictionary have the same length by appending None where necessary.
    """
    max_len = max(len(v) for v in metrics.values())
    for key, value in metrics.items():
        while len(value) < max_len:
            value.append(None)

# Step 5: Visualization
def plot_loss(global_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(global_loss) + 1), global_loss, marker='o')
    plt.title("Global Loss per Communication Round")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

# Step 6: Save Metrics to Excel
# def save_metrics_to_excel(metrics):
#     print("[INFO] Saving metrics to Excel...")
#     metrics_df = pd.DataFrame(metrics)
#     metrics_df.to_excel("federated_learning_metrics.xlsx", index=False)
#     print("[INFO] Metrics saved to 'federated_learning_metrics.xlsx'")

def save_metrics_to_excel(metrics):
    print("[DEBUG] Checking metrics lengths before saving...")
    for key, value in metrics.items():
        print(f"{key}: {len(value)}")

    # Pad metrics to ensure consistent lengths
    max_len = max(len(v) for v in metrics.values())
    for key, value in metrics.items():
        while len(value) < max_len:
            value.append(None)

    # Convert to DataFrame
    print("[INFO] Saving metrics to Excel...")
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_excel("federated_learning_metrics.xlsx", index=False)
    print("[INFO] Metrics saved to 'federated_learning_metrics.xlsx'")


# Main Function (Updated)
def main():
    # Preprocess the dataset
    preprocess_dataset()  # Generates client datasets in "preprocessed_data"

    # Load client data
    clients_data = []
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        file_path = f"preprocessed_data/{season}_data.csv"
        clients_data.append(pd.read_csv(file_path))
        print(f"[INFO] Loaded data for {season}: {clients_data[-1].shape}")


    # [DEBUG] Target statistics for each client
    for i, client_data in enumerate(clients_data):
        print(f"[DEBUG] Client {i + 1} ({season}) Temp Statistics:")
        print(f"Mean: {client_data['Temp'].mean()}")
        print(f"Std Dev: {client_data['Temp'].std()}")
        print(f"Min: {client_data['Temp'].min()}")
        print(f"Max: {client_data['Temp'].max()}")
        print("-------------------------")
    # Initialize global model
    global_model = get_model(MODEL_NAME, input_dim=len(FEATURES), output_dim=1)

    # Get federated algorithm
    federated_algo = get_federated_algo(ALGO_NAME)

    # Loss function
    criterion = nn.MSELoss()  # Use Mean Squared Error for regression tasks

    # Run federated learning
    # global_model, global_loss = federated_learning(
    #     clients_data, global_model, federated_algo, criterion, epochs=EPOCHS, comm_rounds=COMM_ROUNDS
    # )
    global_model, global_loss, global_accuracy = federated_learning(
    clients_data, global_model, federated_algo, criterion, epochs=EPOCHS, comm_rounds=COMM_ROUNDS
    )


    # Save the global model
    torch.save(global_model.state_dict(), f"{MODEL_NAME}_{ALGO_NAME}_global_model.pth")
    print(f"[INFO] Global model saved as {MODEL_NAME}_{ALGO_NAME}_global_model.pth")

    # Plot global loss
    plot_loss(global_loss)

if __name__ == "__main__":
    main()