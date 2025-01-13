from torch.utils.data import Dataset

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the PYTHONPATH or any custom path from .env
python_path = os.getenv("PYTHONPATH")

# Add the path to sys.path if it's set
if python_path and python_path not in os.sys.path:
    os.sys.path.insert(0, python_path)




from src.renewable_energy_price_prediction.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)
