import pytest
import torch
import numpy as np
from typer.testing import CliRunner
from energy.evaluate import app
from energy.model import NeuralNetwork
from pathlib import Path

runner = CliRunner()


@pytest.fixture
def dummy_model(tmp_path):
    input_size = 10
    model = NeuralNetwork(input_size=input_size)
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)
    return model_path


@pytest.fixture
def dummy_processed_data(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    features = torch.randn(100, 10)
    targets = torch.randn(100, 1)
    torch.save(features, processed_dir / "train_features.pt")
    torch.save(targets, processed_dir / "train_targets.pt")
    torch.save(features, processed_dir / "test_features.pt")
    torch.save(targets, processed_dir / "test_targets.pt")
    return processed_dir


def test_evaluate(dummy_model, dummy_processed_data):
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--model-path",
            str(dummy_model),
            "--cfg-path",
            "../../configs",
            "--cfg-name",
            "config.yaml",
        ],
    )
    assert result.exit_code == 0
    assert "Mean Squared Error" in result.output
    assert "R-squared" in result.output
