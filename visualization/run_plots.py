import pandas as pd
from datasets.manager import read_csv
from visualization.plots import (plot_altitude, plot_battery, plot_speed, plot_wind)
data = read_csv("datasets/flight_test.csv")

# plot_altitude(data)
# plot_battery(data)
# plot_speed(data)

# plot_wind(data)
# print("All four graphs generated successfully.")


plot_speed(data.drop(columns=["Speed"]))
# empty_data = pd.DataFrame()
# plot_speed(empty_data)