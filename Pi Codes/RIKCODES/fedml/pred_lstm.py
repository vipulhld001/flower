import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from models import get_model  # Ensure this imports the correct model architecture

# Path to the saved model and input file
saved_model_path = "C:/Users/RIK HALDER/Desktop/federated_3.0/fedml/LSTM_FedAvg_global_model.pth"
#input_file = "data_to_predict.xlsx"
input_file = "new_test_dataset.xlsx"

# Features for prediction
FEATURES = ['dew point', 'humidity', 'precipitation', 'wind speed', 'air pressure']
TARGET = 'Temp'

# Define min and max temperature from the training dataset
min_temp = -15.8
max_temp = 45.0

# Load the pre-trained model architecture and weights
print("[INFO] Loading pre-trained model...")
model = get_model("LSTM", input_dim=len(FEATURES), output_dim=1)  # Recreate the model architecture
model.load_state_dict(torch.load(saved_model_path))  # Load the saved weights
model.eval()  # Set the model to evaluation mode
print("[INFO] Model loaded successfully!")

# Load the dataset for prediction
print("[INFO] Loading dataset for prediction...")
data_to_predict = pd.read_excel(input_file)

# Backup the original Temp column
data_to_predict['Actual_Temp'] = data_to_predict['Temp']  # Ensure 'Actual_Temp' remains untouched

# Load scalers for features
print("[INFO] Scaling features and target...")
feature_scaler = MinMaxScaler()
feature_scaler.fit(data_to_predict[FEATURES])  # Fit scaler to feature columns
data_to_predict[FEATURES] = feature_scaler.transform(data_to_predict[FEATURES])  # Scale features

# Configure target scaler with explicit min and max from training
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit([[min_temp], [max_temp]])  # Fit using training min/max

# Prepare data for prediction
print("[INFO] Preparing data for prediction...")
X = data_to_predict[FEATURES].values.astype(np.float32)
X = torch.tensor(X)

# Reshape for LSTM
X = X.unsqueeze(1)  # Add sequence dimension for LSTM [batch_size, seq_length, input_dim]

# Make predictions
print("[INFO] Making predictions...")
with torch.no_grad():
    predictions = model(X).squeeze().numpy()

# Debugging: Check scaled predictions
print(f"[DEBUG] Scaled Predictions (First 5): {predictions[:5]}")

# Rescale predictions back to original scale
predicted_temp = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Save predictions back to the dataframe
data_to_predict['Predicted_Temp'] = predicted_temp

# Calculate prediction statistics
mse = np.mean((data_to_predict['Actual_Temp'] - data_to_predict['Predicted_Temp']) ** 2)
mae = np.mean(np.abs(data_to_predict['Actual_Temp'] - data_to_predict['Predicted_Temp']))
r_squared = 1 - (np.sum((data_to_predict['Actual_Temp'] - data_to_predict['Predicted_Temp']) ** 2) /
                 np.sum((data_to_predict['Actual_Temp'] - data_to_predict['Actual_Temp'].mean()) ** 2))
tolerance = 0.05  # 5% tolerance for accuracy
accuracy = np.mean(np.abs((data_to_predict['Actual_Temp'] - data_to_predict['Predicted_Temp']) / 
                          data_to_predict['Actual_Temp']) <= tolerance) * 100

# Save the results to an Excel file
output_file = "predicted_results_with_actual_temp_fixed_v2.xlsx"
data_to_predict.to_excel(output_file, index=False)
print(f"[INFO] Predictions saved to {output_file}")

# Save prediction stats to a separate Excel file
stats = {
    "Metric": ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R-squared (R²)", "Accuracy (%)"],
    "Value": [mse, mae, r_squared, accuracy]
}
stats_df = pd.DataFrame(stats)
stats_file = "prediction_stats_fixed_v2.xlsx"
stats_df.to_excel(stats_file, index=False)
print(f"[INFO] Prediction stats saved to {stats_file}")

# Display prediction stats
print("\nPrediction Stats:")
print(stats_df)
