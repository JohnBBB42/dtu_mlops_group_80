import pytorch_lightning as pl
import torch
from torch import nn


class NeuralNetwork(pl.LightningModule):
    def __init__(self, input_size: int, lr: float = 0.01):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

        # Store the learning rate
        self.lr = lr

        # Assuming regression task; change loss if classification or other tasks
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred = y_pred.squeeze(-1)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred = y_pred.squeeze(-1)
        loss = self.loss_fn(y_pred, y)
        self.log("validation_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred = y_pred.squeeze(-1)
        loss = self.loss_fn(y_pred, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Use the stored learning rate
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    # Example usage with dummy data
    input_size = 10  # define input feature size
    model = NeuralNetwork(input_size)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Create dummy input to test forward pass
    dummy_input = torch.randn(1, input_size)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
