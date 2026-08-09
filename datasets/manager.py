from pathlib import Path
import pandas as pd

def read_csv(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return pd.read_csv(path)

data = read_csv("datasets/flight_test.csv")
print(data)


def write_csv(dataframe, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def statistics(dataframe):
    """Return descriptive statistics for an engineering dataset."""
    if dataframe.empty:
        raise ValueError("Dataset cannot be empty.")
    return dataframe.describe()
print(statistics(data))

def filter_data(dataframe, column, minimum):
    if column not in dataframe.columns:
        raise KeyError(f"Column not found: {column}")
    return dataframe[dataframe[column] >= minimum]
high_altitude = filter_data(data, "Altitude", 140)
print(high_altitude)


def export_processed_data(dataframe, file_path):
    """Export processed engineering data to a CSV file."""
    if dataframe.empty:
        raise ValueError("Cannot export an empty dataset.")
    write_csv(dataframe, file_path)

export_processed_data(high_altitude, "reports/processed_flight_test.csv")