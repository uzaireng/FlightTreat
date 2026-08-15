from pathlib import Path
import matplotlib.pyplot as plt
from datasets.manager import get_current_data

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
    plt.xlabel("Altitude (meters)")
    plt.ylabel(column)
    plt.grid(True)
    plt.show()
    # plt.savefig(path, dpi=150, bbox_inches="tight")
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
    plt.show()
    # plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_battery(dataframe, output_path="reports/battery.png"):
    _plot_column(dataframe, "Battery", output_path, "Battery vs Altitude")


def plot_speed(dataframe, output_path="reports/speed.png"):
    _plot_column(dataframe, "Speed", output_path, "Speed vs Altitude")


def plot_wind(dataframe, output_path="reports/wind.png"):
    _plot_column(dataframe, "Wind", output_path, "Wind vs Altitude")

# ==========================================
#            SMART WRAPPER 
# ==========================================

def run_plot_function(plot_func):
    df = get_current_data()
    if df is None:
       print("\n⚠️Firstly, Load the CSV form 'Data Set Manager'!")
       return
    try:
        plot_func(df)
    except Exception as e:
        print(f"Plot Error: {e}")

VISUALIZATION_MENU = {
     "1" : {"title" : "PLOT ALTITUDE GRAPH", "func" : lambda: run_plot_function(plot_altitude)},
     "2" : {"title" : "PLOT BATTERY GRAPH", "func" : lambda: run_plot_function(plot_battery)},
     "3" : {"title" : "PLOT SPEED GRAPH", "func" : lambda: run_plot_function(plot_speed)},
     "4" : {"title" : "PLOT WIND GRAPH", "func" : lambda: run_plot_function(plot_wind)},
}     