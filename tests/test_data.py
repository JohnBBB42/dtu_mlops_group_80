import pytest
import torch
import pandas as pd
from energy.data import main as preprocess_main, load_energy_data, EnergyDataModule


@pytest.fixture
def dummy_processed_data(tmp_path):
    """
    Create dummy processed data files in a temporary directory and return its path.
    """
    # Create dummy data tensors
    features = torch.randn(100, 10)  # 100 samples, 10 features
    targets = torch.randn(100, 1)  # 100 targets
    # Save dummy tensors to temporary directory
    torch.save(features, tmp_path / "train_features.pt")
    torch.save(targets, tmp_path / "train_targets.pt")
    torch.save(features, tmp_path / "test_features.pt")
    torch.save(targets, tmp_path / "test_targets.pt")
    return str(tmp_path)


def test_preprocess_data_creates_pt_files(tmp_path):
    # Set up temporary raw and processed directories
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Create a minimal CSV file with numeric features and target
    csv_content = (
        "f1,f2,target\n"  # header
        "unit1,unit2,unit3\n"  # units row to skip
        "1,2,3\n"
        "4,5,6\n"
    )
    csv_file = raw_dir / "dummy.csv"
    csv_file.write_text(csv_content)

    # Run the data preprocessing
    preprocess_main(raw_dir=str(raw_dir), processed_dir=str(processed_dir))

    # Check that expected .pt files were created
    for fname in ["train_features.pt", "train_targets.pt", "test_features.pt", "test_targets.pt"]:
        file_path = processed_dir / fname
        assert file_path.exists(), f"{fname} was not created."


def test_load_energy_data(dummy_processed_data):
    # Use the dummy processed data from the fixture
    train_dataset, test_dataset = load_energy_data(dummy_processed_data)
    # Validate dataset types
    assert isinstance(train_dataset, torch.utils.data.TensorDataset)
    assert isinstance(test_dataset, torch.utils.data.TensorDataset)
    # Validate shapes of loaded tensors
    train_features, train_targets = train_dataset.tensors
    test_features, test_targets = test_dataset.tensors
    assert train_features.shape == (100, 10)
    assert train_targets.shape == (100, 1)
    assert test_features.shape == (100, 10)
    assert test_targets.shape == (100, 1)


def test_energy_datamodule_setup(tmp_path):
    # Set up temporary raw and processed directories
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    # Create a minimal CSV file
    csv_content = (
        "f1,f2,target\n"  # header
        "unit1,unit2,unit3\n"  # units row
        "1,2,3\n"
        "4,5,6\n"
    )
    dummy_csv = raw_dir / "dummy.csv"
    dummy_csv.write_text(csv_content)

    # Read CSV to determine feature count dynamically
    df = pd.read_csv(dummy_csv, skiprows=[1])
    feature_count = df.shape[1] - 1  # subtract target column

    # Preprocess data to create .pt files in our temporary processed_dir
    preprocess_main(raw_dir=str(raw_dir), processed_dir=str(processed_dir))

    # Initialize the EnergyDataModule with the temporary processed_dir
    dm = EnergyDataModule(data_dir=str(processed_dir), batch_size=2)
    dm.setup(stage="fit")

    # Get dataloaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Fetch one batch from train_loader and validate shapes
    for batch in train_loader:
        x, y = batch
        assert x.shape[1] == feature_count
        # Handle both 1D and 2D target tensors
        if y.ndim == 2:
            assert y.shape[1] == 1
        else:
            assert y.ndim == 1
        break

    # Validate that validation and test loaders yield batches with correct shapes
    for loader in [val_loader, test_loader]:
        for batch in loader:
            x, y = batch
            assert x.shape[1] == feature_count
            if y.ndim == 2:
                assert y.shape[1] == 1
            else:
                assert y.ndim == 1
            break
