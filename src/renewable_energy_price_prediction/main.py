import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from train import train_simple_model, train_complex_model
from evaluate import evaluate_simple_model, evaluate_complex_model


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


if __name__ == "__main__":
    # Path to the dataset
    file_path = "data\processed\processed_data.csv"

    # Load and preprocess data
    X, y = load_data(file_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the simple model
    simple_model = train_simple_model(X_train, y_train)
    evaluate_simple_model(simple_model, X_test, y_test)

    # Train and evaluate the complex model
    input_size = X_train.shape[1]
    complex_model = train_complex_model(X_train, y_train, input_size)
    evaluate_complex_model(complex_model, X_test, y_test)
