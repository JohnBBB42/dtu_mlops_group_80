import pytest
import torch
from energy.model import NeuralNetwork

@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    # Define an input size that matches the model's expected input dimension
    input_size = 10
    model = NeuralNetwork(input_size=input_size)
    # Create dummy input with shape [batch_size, input_size]
    x = torch.randn(batch_size, input_size)
    y = model(x)
    # The model outputs a tensor of shape [batch_size, 1] for regression
    assert y.shape == (batch_size, 1)

def test_training_step():
    input_size = 10
    model = NeuralNetwork(input_size=input_size)
    # Override the log method to avoid errors during tests
    model.log = lambda *args, **kwargs: None
    # Create a dummy batch of inputs and regression targets
    images = torch.randn(4, input_size)
    targets = torch.randn(4, 1)  # Targets for regression
    batch = (images, targets)

    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None, "training_step did not return a loss."
    assert loss.requires_grad, "loss should require grad for backprop."

def test_validation_step():
    input_size = 10
    model = NeuralNetwork(input_size=input_size)
    model.log = lambda *args, **kwargs: None
    images = torch.randn(4, input_size)
    targets = torch.randn(4, 1)
    batch = (images, targets)

    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None, "validation_step did not return a loss."

def test_test_step():
    input_size = 10
    model = NeuralNetwork(input_size=input_size)
    model.log = lambda *args, **kwargs: None
    images = torch.randn(4, input_size)
    targets = torch.randn(4, 1)
    batch = (images, targets)

    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None, "test_step did not return a loss."
