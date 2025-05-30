import flwr as fl


NUM_ROUNDS = 20  # Number of rounds to train
# Start Flower server
fl.server.start_server(
  server_address="192.168.10.103:8080",
  config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
)