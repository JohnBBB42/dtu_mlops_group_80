"""
Train and evaluate models using Hydra configuration management.
"""

import logging
import os
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from sklearn.metrics import mean_squared_error, r2_score
from model import get_linear_regression_model, NeuralNetwork
from evaluate import evaluate_simple_model, evaluate_complex_model
# Load and preprocess data
from utils import load_data, split_data

log = logging.getLogger(__name__)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    @hydra.main(version_base="1.1", config_path="../../configs", config_name="config.yaml")
    def train(config) -> None:
        """
        Train and evaluate models based on configuration.
        Supports both simple (linear regression) and complex (neural network) models.
        """
        log.info(f"Configuration: \n{OmegaConf.to_yaml(config)}")
        hparams = config.experiment

        
        X, y = load_data(hparams["dataset_path"])
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=hparams["test_size"])

        if config.mode == "simple":
            log.info("Training Simple Model (Linear Regression)...")
            model = get_linear_regression_model()
            model.fit(X_train, y_train)
            log.info("Evaluating Simple Model...")
            evaluate_simple_model(model, X_test, y_test)
            #save_model(model, "simple_model.pkl")
        elif config.mode == "complex":
            log.info("Training Complex Model (Neural Network)...")
            input_size = X_train.shape[1]

            # Prepare data for PyTorch
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
            train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)

            model = NeuralNetwork(input_size)
            criterion = nn.MSELoss()
            optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Training loop
            model.train()
            for epoch in range(hparams["n_epochs"]):
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                log.info(f"Epoch {epoch + 1}/{hparams['n_epochs']}, Loss: {epoch_loss:.4f}")

            log.info("Evaluating Complex Model...")
            evaluate_complex_model(model, X_test, y_test)
        else:
            log.error("Invalid mode. Use 'simple' or 'complex'.")
            return

        log.info("Training complete!")

        # save weights
        torch.save(model, f"{os.getcwd()}/trained_model.pt")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
        prof.export_chrome_trace("trace.json")

if __name__ == "__main__":
    train()
