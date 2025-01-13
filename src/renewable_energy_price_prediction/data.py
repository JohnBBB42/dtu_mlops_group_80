from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
import typer


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        """
        Initialize the dataset with the raw data path.
        Args:
            raw_data_path (Path): Path to the folder containing raw data files.
        """
        self.data_path = raw_data_path
        self.data_files = list(raw_data_path.glob("*.csv"))  # Assuming CSV files
        if not self.data_files:
            raise ValueError(f"No CSV files found in {raw_data_path}")

    def __len__(self) -> int:
        """Return the number of data files."""
        return len(self.data_files)

    def __getitem__(self, index: int) -> pd.DataFrame:
        """Return a DataFrame for a specific raw file."""
        file_path = self.data_files[index]
        return pd.read_csv(file_path)

    def preprocess(self, output_folder: Path) -> None:
        """
        Preprocess the raw data and save it to the output folder as a single file.
        Args:
            output_folder (Path): Path to save the processed data.
        """
        output_folder.mkdir(parents=True, exist_ok=True)
        all_data = []

        print("Merging data...")
        for file_path in self.data_files:
            print(f"Processing file: {file_path}")
            # Read the CSV file, enforcing dtypes (float for all except 'Date')
            df = pd.read_csv(file_path)
            df.drop(index=df.index[0], axis=0, inplace=True)

            # Convert all columns to float except for 'Date'
            df = df.astype({col: 'float' for col in df.columns if col != 'Date (GMT+1)'})
            all_data.append(df)

        # Concatenate all data into a single DataFrame
        merged_data = pd.concat(all_data, ignore_index=True)

        # Save the merged DataFrame to the output folder
        output_file = output_folder / "processed_data.csv"
        merged_data.to_csv(output_file, index=False)
        print(f"Processed data saved to: {output_file}")



def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    """
    Preprocess raw data files and save the processed data to the output folder.
    Args:
        raw_data_path (Path): Path to the folder containing raw data files.
        output_folder (Path): Path to the folder to save processed data.
    """
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
