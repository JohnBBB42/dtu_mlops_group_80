import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from omegaconf import DictConfig
from typer.testing import CliRunner
from energy.evaluate import app

runner = CliRunner()


@pytest.fixture
def mock_config():
    return DictConfig({"hyperparameters": {"batch_size": 4, "lr": 0.001}})


@pytest.fixture
def mock_test_dataset():
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    targets = torch.tensor([0.5, 1.5], dtype=torch.float32)
    return [(features[i], targets[i]) for i in range(2)]


@pytest.fixture
def mock_model():
    model = Mock()
    model.eval = Mock()
    predictions = torch.tensor([0.4, 1.6], dtype=torch.float32)
    model.return_value = predictions
    return model


def test_cli_evaluate(tmp_path):
    model_path = tmp_path / "model.pth"
    model_path.touch()

    result = runner.invoke(
        app,
        [
            "--model-path",
            str(model_path),
            "--cfg-path",
            "configs",
            "--cfg-name",
            "config.yaml",
        ],
    )
    assert result.exit_code == 0


@patch("evaluate.hydra.initialize")
@patch("evaluate.hydra.compose")
@patch("evaluate.EnergyDataModule")
@patch("evaluate.NeuralNetwork")
@patch("torch.load")
def test_model_evaluation(
    mock_torch_load,
    mock_nn,
    mock_data_module,
    mock_compose,
    mock_initialize,
    mock_config,
    mock_test_dataset,
    capsys,
):
    # Setup mocks
    mock_compose.return_value = mock_config
    mock_data_module.return_value.test_dataset = mock_test_dataset

    # Run evaluation
    result = runner.invoke(
        app,
        [
            "--model-path",
            "dummy_model.pth",
            "--cfg-path",
            "configs",
            "--cfg-name",
            "config.yaml",
        ],
    )

    # Verify outputs
    captured = capsys.readouterr()
    assert "Complex Model - Neural Network" in captured.out
    assert "Mean Squared Error:" in captured.out
    assert "R-squared:" in captured.out


def test_data_processing(mock_test_dataset):
    X_list, y_list = [], []
    for features, target in mock_test_dataset:
        X_list.append(features.numpy())
        y_list.append(target.numpy())

    X_test = np.array(X_list, dtype=np.float32)
    y_test = np.array(y_list, dtype=np.float32)

    assert X_test.shape == (2, 2)
    assert y_test.shape == (2,)

    # Verify data matches original tensors
    np.testing.assert_array_almost_equal(X_test, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
