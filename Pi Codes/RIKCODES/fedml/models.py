import torch
import torch.nn as nn

# MLP Model (Simplified)
# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MLP, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 32),  # Reduced neurons
#             nn.ReLU(),
#             nn.Linear(32, output_dim)
#         )
    
#     def forward(self, x):
#         return self.fc(x)
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),  # Reduced neurons
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Add dropout
            nn.Linear(32, 16),  # Another layer with fewer neurons
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)



# LSTM Model
# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)  # hn is the final hidden state
#         return self.fc(hn[-1])
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM returns output and (hidden_state, cell_state)
        _, (hn, _) = self.lstm(x)  # hn is the final hidden state [num_layers, batch_size, hidden_dim]
        hn = hn[-1]  # Take the last layer's hidden state [batch_size, hidden_dim]
        output = self.fc(hn)  # Pass through fully connected layer [batch_size, output_dim]
        return output



def calculate_conv_output_size(input_size, conv_layers):
    # Ensure the input has a sufficiently large sequence length
    dummy_input = torch.zeros(1, *input_size)  # Batch size = 1
    if input_size[1] == 1:  # If sequence length = 1, use a larger dummy sequence for calculation
        dummy_input = torch.zeros(1, input_size[0], 10)  # Use a sequence length of 10
    with torch.no_grad():
        output = conv_layers(dummy_input)
        return output.numel()


# CNN Model
class CNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1) if input_channels > 1 else nn.Identity(),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1) if input_channels > 1 else nn.Identity()
        )
        
        # Dynamically calculate the flattened size
        conv_output_size = calculate_conv_output_size((input_channels, 1), self.conv)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 64),  # Use calculated size
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        print(f"[DEBUG] Input to CNN: {x.shape}")
        x = self.conv(x)
        print(f"[DEBUG] After Conv Layers: {x.shape}")
        x = self.fc(x)
        print(f"[DEBUG] After FC: {x.shape}")
        return x


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        
        # Ensure d_model is divisible by num_heads
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads}).")
        
        self.embedding = nn.Linear(input_dim, num_heads * (input_dim // num_heads))  # Embed to match d_model
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_heads * (input_dim // num_heads),  # Embedding dimension
            nhead=num_heads,  # Number of attention heads
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.fc = nn.Linear(num_heads * (input_dim // num_heads), output_dim)  # Fully connected output layer
    
    def forward(self, x):
        x = self.embedding(x)  # Map input to embedding space
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Aggregate sequence info (e.g., mean pooling)
        return self.fc(x)

# Model Selector
def get_model(model_name, input_dim, output_dim, **kwargs):
    print(f"[INFO] Initializing model: {model_name}")
    if model_name == "MLP":
        return MLP(input_dim, output_dim)
    elif model_name == "LSTM":
        return LSTMModel(input_dim, kwargs.get('hidden_dim', 64), output_dim, kwargs.get('num_layers', 1))
    elif model_name == "CNN":
        return CNN(input_dim, output_dim)
    elif model_name == "Transformer":
        num_heads = kwargs.get('num_heads', 4)
        # Adjust input_dim if needed to make it divisible by num_heads
        if input_dim % num_heads != 0:
            input_dim = (input_dim // num_heads) * num_heads  # Round down to nearest multiple
            print(f"[INFO] Adjusted input_dim to {input_dim} to match num_heads.")
        return TransformerModel(input_dim, num_heads, kwargs.get('num_layers', 2), output_dim)
    else:
        raise ValueError(f"[ERROR] Unknown model name: {model_name}")
