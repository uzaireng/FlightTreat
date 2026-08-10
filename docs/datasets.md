
---

# Part 4 — `datasets.md`

Create:

```markdown
# Engineering Dataset Manager

## Purpose

The dataset module provides basic tools for managing engineering data
using Pandas.

It forms the foundation for future flight-test and simulation datasets.

## Current Dataset

V0.1 includes:

```text
datasets/flight_test.csv

| Column      | Description             |
| ----------- | ----------------------- |
| Altitude    | UAV flight altitude     |
| Battery     | Remaining battery level |
| Speed       | UAV speed               |
| Wind        | Wind measurement        |
| Temperature | Recorded temperature    |

Current Capabilities
Read CSV

Loads an engineering CSV file into a Pandas DataFrame.

Write CSV

Writes a DataFrame to a CSV file.

Statistics

Generates descriptive statistics including:

count
mean
standard deviation
minimum
maximum
quartiles
Filtering

Allows rows to be selected according to a minimum value for a specified
column.

Export

Processed datasets can be exported as CSV files.

CSV Dataset
    ↓
Pandas DataFrame
    ↓
Analysis
    ↓
Filtering
    ↓
Processed Dataset
    ↓
CSV Export

Future Development

Future versions may add:

Units
Timestamps
GPS coordinates
Air density
Pressure
Lift
Drag
Reynolds number
Simulation results