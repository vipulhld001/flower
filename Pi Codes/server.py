import flwr as fl

# Start Flower server
fl.server.start_server(
  server_address="192.168.10.103:8080",
  config=fl.server.ServerConfig(num_rounds=20),
)