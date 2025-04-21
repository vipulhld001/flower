import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Step 1: Load the dataset
def load_dataset(file_path):
    try:
        # Read dataset
        print("[INFO] Loading dataset...")
        df = pd.read_excel(file_path)  # Change to `pd.read_excel()` for Excel files
        print("[INFO] Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return None

# Step 2: Add a season column
def add_season_column(df):
    print("[INFO] Extracting season information from 'Date' column...")
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                      'Spring' if x in [3, 4, 5] else
                                      'Summer' if x in [6, 7, 8] else
                                      'Autumn')
    if df['Season'].isnull().any():
        print("[WARNING] Some rows have invalid dates. Check the data!")
    print("[INFO] Season column added successfully!")
    return df

# Step 3: Split data by season (client-wise)
def split_by_season(df):
    print("[INFO] Splitting dataset into clients by season...")
    clients = {}
    for season in df['Season'].unique():
        clients[season] = df[df['Season'] == season].copy()
        print(f"[INFO] {season} data shape: {clients[season].shape}")
    return clients

# Step 4: Normalize features
def normalize_features(clients, features_to_normalize):
    print("[INFO] Normalizing features...")
    scaler = MinMaxScaler()
    normalized_clients = {}
    for season, data in clients.items():
        data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])
        normalized_clients[season] = data
        print(f"[INFO] Normalized data for {season}")
    return normalized_clients

# Step 5: Save preprocessed data
def save_clients(clients, output_dir):
    print(f"[INFO] Saving preprocessed data to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    for season, data in clients.items():
        file_path = os.path.join(output_dir, f"{season}_data.csv")
        data.to_csv(file_path, index=False)
        print(f"[INFO] Saved {season} data to {file_path}")

# Main Preprocessing Flow
def main():
    file_path = "finals.xlsx"  # Replace with your actual dataset file
    output_dir = "preprocessed_data"
    features_to_normalize = ['Temp', 'dew point', 'humidity', 'precipitation', 
                             'wind speed', 'air pressure']

    # Load dataset
    df = load_dataset(file_path)
    if df is None:
        return

    # Add season column
    df = add_season_column(df)

    # Split dataset by season
    clients = split_by_season(df)

    # Normalize features
    clients = normalize_features(clients, features_to_normalize)

    # Save preprocessed data
    save_clients(clients, output_dir)

    print("[INFO] Preprocessing complete!")

if __name__ == "__main__":
    main()
