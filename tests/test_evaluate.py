import pytest
from unittest.mock import patch, Mock
from pathlib import Path
from omegaconf import DictConfig
from typer.testing import CliRunner
import torch
from energy.evaluate import app

runner = CliRunner()


def test_cli_evaluate(tmp_path):
    mock_dataset = [(torch.tensor([1.0, 2.0]), torch.tensor(0.5))]

    with (
        patch("energy.evaluate.hydra.initialize") as mock_init,
        patch("energy.evaluate.hydra.compose") as mock_compose,
        patch("energy.evaluate.EnergyDataModule") as mock_dm,
        patch("energy.evaluate.NeuralNetwork") as mock_nn,
        patch("torch.load"),
    ):

        # Configure mocks
        mock_compose.return_value = DictConfig(
            {"hyperparameters": {"batch_size": 4, "lr": 0.001}}
        )

        mock_dm.return_value.test_dataset = mock_dataset
        mock_dm.return_value.setup = Mock()

        model_path = tmp_path / "model.pth"
        model_path.touch()

        result = runner.invoke(
            app,
            [
                "--model-path",
                str(model_path),
                "--cfg-path",
                str(tmp_path),
                "--cfg-name",
                "config.yaml",
            ],
        )

        assert result.exit_code == 0
