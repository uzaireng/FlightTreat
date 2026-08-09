from pathlib import Path
import matplotlib.pyplot as plt
# import pandas as pd

def _plot_column(dataframe, column, output_path, title):
    """Plot one dataset column against altitude."""
    required_columns = {"Altitude", column}

    if dataframe.empty:
            raise ValueError("Dataset cannot be empty.")

    if not required_columns.issubset(dataframe.columns):
        missing = required_columns - set(dataframe.columns)
        raise KeyError(f"Missing column: {missing}")

    

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(dataframe["Altitude"], dataframe[column])

    plt.title(title)
    plt.xlabel("Altitude")
    plt.ylabel(column)
    plt.grid(True)

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()



def plot_altitude(dataframe, output_path="reports/altitudes.png"):
    """Generate an altitude visualization."""
    if "Altitude" not in dataframe.columns:
        raise KeyError("Missing column: Altitude")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(dataframe["Altitude"])
    plt.title("UAV Altitude")
    plt.xlabel("Flight Sample")
    plt.ylabel("Altitude")
    plt.grid(True)
    # plt.show()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_battery(dataframe, output_path="reports/battery.png"):
    _plot_column(dataframe, "Battery", output_path, "Battery vs Altitude")


def plot_speed(dataframe, output_path="reports/speed.png"):
    _plot_column(dataframe, "Speed", output_path, "Speed vs Altitude")


def plot_wind(dataframe, output_path="reports/wind.png"):
    _plot_column(dataframe, "Wind", output_path, "Wind vs Altitude")

