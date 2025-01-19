import os
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

def main(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> None:
    """Process raw CSV data and save processed tensors."""
    os.makedirs(processed_dir, exist_ok=True)
    all_dfs = []

    # Read and merge all CSV files from raw_dir, skipping the units row
    for file_name in os.listdir(raw_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(raw_dir, file_name)
            # Skip second row which contains units
            df = pd.read_csv(file_path, skiprows=[1], low_memory=False)
            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Separate features and target: assume last column is target
    features_df = merged_df.iloc[:, :-1]
    target_series = merged_df.iloc[:, -1]

    # Select only numeric columns for features
    numeric_features = features_df.select_dtypes(include=[np.number])
    numeric_target = pd.to_numeric(target_series, errors='coerce')

    # Combine numeric features and target
    combined = pd.concat([numeric_features, numeric_target.rename("target")], axis=1)

    # Drop rows where the target is NaN
    combined = combined.dropna(subset=["target"])

    # Fill any remaining NaN values in features
    combined.fillna(method='ffill', inplace=True)
    combined.fillna(method='bfill', inplace=True)

    if combined.empty:
        raise ValueError("No valid numeric data after preprocessing. Check your raw data and conversion steps.")

    # Separate cleaned features and target
    features = combined.drop(columns=["target"]).values
    targets = combined["target"].values

    # Normalize features column-wise
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    # Convert to tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)

    # Split into train and test sets
    train_features, test_features, train_targets, test_targets = train_test_split(
        features_tensor, targets_tensor, test_size=0.2, random_state=42
    )

    # Save processed tensors
    torch.save(train_features, os.path.join(processed_dir, "train_features.pt"))
    torch.save(train_targets, os.path.join(processed_dir, "train_targets.pt"))
    torch.save(test_features, os.path.join(processed_dir, "test_features.pt"))
    torch.save(test_targets, os.path.join(processed_dir, "test_targets.pt"))
    print(f"Data preprocessed and saved to {processed_dir}")

def load_energy_data(processed_dir: str = "data/processed") -> tuple[TensorDataset, TensorDataset]:
    """Load processed tensors and return TensorDataset objects for train and test."""
    train_features = torch.load(os.path.join(processed_dir, "train_features.pt"), weights_only=True)
    train_targets = torch.load(os.path.join(processed_dir, "train_targets.pt"), weights_only=True)
    test_features = torch.load(os.path.join(processed_dir, "test_features.pt"), weights_only=True)
    test_targets = torch.load(os.path.join(processed_dir, "test_targets.pt"), weights_only=True)

    train_dataset = TensorDataset(train_features, train_targets)
    test_dataset = TensorDataset(test_features, test_targets)
    return train_dataset, test_dataset

class EnergyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/processed", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        """Load datasets for training and testing."""
        if stage == "fit" or stage is None:
            self.train_dataset, self.test_dataset = load_energy_data(self.data_dir)
        if stage == "test" or stage is None:
            _, self.test_dataset = load_energy_data(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # For demonstration, using test dataset as validation
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

if __name__ == "__main__":
    main()
