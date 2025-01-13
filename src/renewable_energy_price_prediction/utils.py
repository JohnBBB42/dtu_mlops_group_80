import pandas as pd
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
def load_data(file_path):
    # Load the CSV data
    df = pd.read_csv(file_path)
    

    # Drop missing values (if any)
    df = df.fillna(0)
    df.reset_index(drop=True, inplace=True)

    # Extract features (X) and target (y)
    X = df.drop(columns=["Date (GMT+1)", "Day Ahead Auction (DE-LU)"]).values  # All columns except 'Date' and 'Price (EUR/MWh)'
    y = df['Day Ahead Auction (DE-LU)'].values    # Target: 'Price (EUR/MWh)'
    print(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Target has {len(y)} samples")

    return X, y

def split_data(X, y, test_size=0.2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"Training data: {X_train.shape} | Testing data: {X_test.shape}")
    return X_train, X_test, y_train, y_test