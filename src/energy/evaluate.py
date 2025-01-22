# evaluate.py
import typer
import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import mean_squared_error, r2_score
from energy.data import EnergyDataModule
from energy.model import NeuralNetwork

app = typer.Typer()


@app.command()
def evaluate(
    model_path: str = typer.Option("models/model.pth", help="Path to the saved model file."),
    cfg_path: str = typer.Option("../../configs", help="Path to Hydra configuration directory."),
    cfg_name: str = typer.Option("config.yaml", help="Name of the Hydra configuration file."),
):
    """
    Evaluate a trained neural network model using specified configuration.
    """

    def evaluate_neural_network(model, X_test, y_test):
        # Evaluate the neural network
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

        print("\nComplex Model - Neural Network")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R-squared: {r2:.2f}")

    def hydra_main(cfg: DictConfig) -> None:
        print("Hydra Configuration:")
        print(OmegaConf.to_yaml(cfg))

        # Use hyperparameters from config
        effective_batch_size = cfg.hyperparameters.batch_size

        # Setup Data
        data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
        data_module = EnergyDataModule(data_dir=str(data_dir), batch_size=effective_batch_size)
        data_module.setup("test")

        test_dataset = data_module.test_dataset

        # Build X_test and y_test as numpy arrays
        X_list, y_list = [], []
        for features, target in test_dataset:
            X_list.append(features.numpy())
            y_list.append(target.numpy())
        X_test = np.array(X_list, dtype=np.float32)
        y_test = np.array(y_list, dtype=np.float32)

        # Infer input size from the dataset
        sample_features, _ = test_dataset[0]
        input_size = sample_features.shape[0]

        # Initialize model
        model = NeuralNetwork(input_size=input_size, lr=cfg.hyperparameters.lr)
        model.load_state_dict(torch.load(model_path))

        # Evaluate the model
        evaluate_neural_network(model, X_test, y_test)

    # Hydra setup: Avoid parsing Typer arguments as Hydra arguments
    with hydra.initialize(config_path=cfg_path):
        cfg = hydra.compose(config_name=cfg_name)
        hydra_main(cfg)


if __name__ == "__main__":
    app()
