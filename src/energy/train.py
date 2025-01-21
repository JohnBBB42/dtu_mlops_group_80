import typer
import pytorch_lightning as pl
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from omegaconf import OmegaConf
from energy.model import NeuralNetwork
from energy.evaluate import evaluate_neural_network
import wandb
import logging

# Load and preprocess data
from energy.data import EnergyDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from typing_extensions import Annotated
from torch.profiler import profile, ProfilerActivity

# Initialize Typer app
app = typer.Typer()

# Initialize WandB
run = wandb.init(project="energy_prediction", job_type="training")
logger = pl.loggers.WandbLogger(project="lightning_energy")
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    @app.command()
    def train(
        lr: Annotated[float, typer.Option(help="Learning rate for training.")] = None,
        batch_size: Annotated[int, typer.Option(help="Batch size for training.")] = None,
        epochs: Annotated[int, typer.Option(help="Number of training epochs.")] = None,
        cfg_path: Annotated[str, typer.Option(help="Path to Hydra configuration directory.")] = "../../configs",
        cfg_name: Annotated[str, typer.Option(help="Name of the Hydra configuration file.")] = "config.yaml",
    ):
        """
        Train a neural network model with specified hyperparameters.
        """

        def hydra_main(cfg: DictConfig) -> None:
            # Combine Hydra configuration with CLI arguments
            print("Hydra Configuration:")
            print(OmegaConf.to_yaml(cfg))

            # Combine CLI arguments with Hydra configuration
            effective_lr = lr if lr is not None else cfg.hyperparameters.lr
            effective_batch_size = batch_size if batch_size is not None else cfg.hyperparameters.batch_size
            effective_epochs = epochs if epochs is not None else cfg.hyperparameters.n_epochs

            # Setup Data
            data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
            data_module = EnergyDataModule(data_dir=str(data_dir), batch_size=effective_batch_size)
            data_module.setup("fit")

            # Setup Model
            sample_features, _ = data_module.train_dataset[0]
            input_size = sample_features.shape[0]
            model = NeuralNetwork(input_size=input_size, lr=effective_lr)  # Pass learning rate to the model

            # Callbacks
            early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
            checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")

            # Trainer
            trainer = pl.Trainer(
                default_root_dir="my_logs_dir",
                max_epochs=effective_epochs,
                callbacks=[early_stopping_callback, checkpoint_callback],
                logger=logger,
                log_every_n_steps=1,
            )

            # Training
            trainer.fit(model, datamodule=data_module)
            trainer.test(model, datamodule=data_module)

            log.info("Evaluating Complex Model...")

            data_module.setup("test")
            test_dataset = data_module.test_dataset

            # Build X_test, y_test as numpy arrays
            X_list, y_list = [], []
            for features, target in test_dataset:
                X_list.append(features.numpy())  # convert from tensor to numpy
                y_list.append(target.numpy())

            X_test = np.array(X_list, dtype=np.float32)
            y_test = np.array(y_list, dtype=np.float32)

            evaluate_neural_network(model, X_test, y_test)

            log.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            # logger.experiment.log({"profiler": prof.key_averages().table(sort_by="cpu_time_total").to_json()})
            # log.info("Evaluating Complex Model...")
            # evaluate_complex_model(model, X_test, y_test)
            # log.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            log.info("Training complete!")

            # Save Model
            torch.save(model.state_dict(), "model.pth")
            artifact = wandb.Artifact(name="example_artifact", type="model")
            artifact.add_file("model.pth")
            run.log_artifact(artifact)

        # Hydra setup: Avoid parsing `typer` arguments
        with hydra.initialize(config_path=cfg_path):
            cfg = hydra.compose(config_name=cfg_name)
            hydra_main(cfg)


if __name__ == "__main__":
    app()
