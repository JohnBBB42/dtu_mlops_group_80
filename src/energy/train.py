import typer
import pytorch_lightning as pl
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from energy.model import NeuralNetwork
from energy.data import EnergyDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import wandb

# Initialize Typer app
app = typer.Typer()

# Initialize WandB
run = wandb.init(project="energy_prediction", job_type="training")
logger = pl.loggers.WandbLogger(project="lightning_energy")

@app.command()
def train(
    lr: float = typer.Option(0.001, help="Learning rate for training."),
    batch_size: int = typer.Option(32, help="Batch size for training."),
    epochs: int = typer.Option(10, help="Number of training epochs."),
    hydra_cfg_path: str = typer.Option("../../configs", help="Path to Hydra configuration directory."),
    hydra_cfg_name: str = typer.Option("config.yaml", help="Name of the Hydra configuration file."),
):
    """
    Train a neural network model with specified hyperparameters.
    """

    @hydra.main(version_base="1.1", config_path=hydra_cfg_path, config_name=hydra_cfg_name)
    def hydra_main(cfg: DictConfig) -> None:
        # Combine Hydra configuration with CLI arguments
        print("Hydra Configuration:")
        print(OmegaConf.to_yaml(cfg))

        # Log CLI arguments
        print(f"CLI Arguments: lr={lr}, batch_size={batch_size}, epochs={epochs}")

        # Setup Data
        data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
        data_module = EnergyDataModule(data_dir=str(data_dir), batch_size=batch_size)
        data_module.setup("fit")

        # Setup Model
        sample_features, _ = data_module.train_dataset[0]
        input_size = sample_features.shape[0]
        model = NeuralNetwork(input_size=10, lr=lr)  # Pass learning rate to the model
        print(f"Initialized model with input size: {input_size}")

        # Callbacks
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
        checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")

        # Trainer
        trainer = pl.Trainer(
            default_root_dir="my_logs_dir",
            max_epochs=epochs,
            callbacks=[early_stopping_callback, checkpoint_callback],
            profiler="simple",
            logger=logger,
            log_every_n_steps=1,
        )

        # Training
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

        # Save Model
        torch.save(model.state_dict(), "model.pth")
        artifact = wandb.Artifact(name="example_artifact", type="model")
        artifact.add_file("model.pth")
        run.log_artifact(artifact)

    # Run Hydra main with the passed configuration
    hydra_main()

if __name__ == "__main__":
    app()
