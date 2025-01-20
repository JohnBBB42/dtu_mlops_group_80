"""
Train and evaluate models using Hydra configuration management.
"""

import pytorch_lightning as pl
import logging
from pathlib import Path
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from energy.model import NeuralNetwork
from energy.evaluate import evaluate_neural_network

# Load and preprocess data
from energy.data import EnergyDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Dynamically determine the path to the configs directory
script_dir = Path(__file__).parent  # src/renewable_energy_price_prediction/
project_root = script_dir.parent.parent  # dtu_mlops_group_80/
config_dir = project_root / "configs"  # dtu_mlops_group_80/configs

# Determine absolute path to the processed data directory
data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"

log = logging.getLogger(__name__)
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True, profile_memory=True
) as prof:

    @hydra.main(version_base="1.1", config_path=str(config_dir), config_name="config.yaml")
    def main(config) -> None:
        """
        Train and evaluate models based on configuration.
        Supports both simple (linear regression) and complex (neural network) models.
        """
        log.info(f"Configuration: \n{OmegaConf.to_yaml(config)}")
        hparams = config.hyperparameters

        data_module = EnergyDataModule(data_dir=str(data_dir), batch_size=hparams.batch_size)
        data_module.setup("fit")  # Load the datasets for 'fit' stage

        sample_features, _ = data_module.train_dataset[0]
        input_size = sample_features.shape[0]
        model = NeuralNetwork(
            input_size=input_size,
            hidden_units=hparams.hidden_units,
            lr=hparams.learning_rate
        )

        print(f"Initialized model with input size: {input_size}")

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping.patience,
            verbose=True,
            mode="min")

        checkpoint_callback = ModelCheckpoint(
            dirpath="./models",
            monitor="val_loss",
            mode="min")

        trainer = pl.Trainer(
            default_root_dir="my_logs_dir",
            max_epochs=hparams.max_epochs,
            limit_train_batches=config.trainer.limit_train_batches,
            callbacks=[early_stopping_callback, checkpoint_callback],
            profiler="simple",
            #logger=pl.loggers.WandbLogger(project="lightning_energy"),
            logger=False,
        )

        log.info("Training Neural Network...")
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
        log.info("Training complete!")


if __name__ == "__main__":
    main()
