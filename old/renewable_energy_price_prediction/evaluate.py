import torch
from sklearn.metrics import mean_squared_error, r2_score


# Evaluate Simple Model (Linear Regression)
def evaluate_simple_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Simple Model - Linear Regression")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")


# Evaluate Complex Model (Neural Network)
def evaluate_complex_model(model, X_test, y_test):
    # Convert test data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Evaluate the neural network
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

    print("\nComplex Model - Neural Network")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")
