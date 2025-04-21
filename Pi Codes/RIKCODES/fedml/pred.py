from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np
from models import get_model

# Load the saved model
saved_model_path = "MLP_FedAvg_global_model.pth"
model = get_model("MLP", input_dim=5, output_dim=1)  # Adjust input_dim if needed
model.load_state_dict(torch.load(saved_model_path))
model.eval()

# Load the test dataset
test_data_path = "data_to_predict.xlsx"  # Replace with your test dataset path
test_data = pd.read_excel(test_data_path)

# Define features and target
FEATURES = ['dew point', 'humidity', 'precipitation', 'wind speed', 'air pressure']
TARGET = 'Temp'

# Reconstruct the scaler manually for Temp
# Replace these values with the actual min and max of the Temp column from training
TEMP_MIN = -15.8  # Minimum Temp during training
TEMP_MAX = 45.0  # Maximum Temp during training

# Normalize the features (you can reuse the MinMaxScaler logic for features if required)
scaler = MinMaxScaler()
test_data[FEATURES] = scaler.fit_transform(test_data[FEATURES])

# Prepare the data for prediction
test_features = test_data[FEATURES].values.astype(np.float32)
test_features = torch.tensor(test_features)

# Make predictions
with torch.no_grad():
    predictions = model(test_features).squeeze().numpy()

# Inverse transform predictions manually for Temp
predictions_original = predictions * (TEMP_MAX - TEMP_MIN) + TEMP_MIN

# Add predictions to the dataframe
test_data['Predicted_Temp'] = predictions_original

# Calculate prediction statistics
mse = np.mean((test_data[TARGET] - test_data['Predicted_Temp']) ** 2)
mae = np.mean(np.abs(test_data[TARGET] - test_data['Predicted_Temp']))
accuracy_within_5_percent = (
    np.sum(np.abs(test_data[TARGET] - test_data['Predicted_Temp']) <= 0.10 * test_data[TARGET])
    / len(test_data)
) * 100

# Save the results
test_data.to_excel("predictions_with_stats.xlsx", index=False)
print("[INFO] Predictions saved to 'predictions_with_stats.xlsx'")

# Print prediction stats
print(f"Prediction Stats:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Accuracy (within ±5% tolerance): {accuracy_within_5_percent:.2f}%")
