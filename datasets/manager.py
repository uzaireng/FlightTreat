from pathlib import Path
import pandas as pd


def _validate_number(value, name):
    """Validate that a input is a valid number."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    # if value <=0:
    #     raise ValueError(f"{name} must be greater than zero.")

#----------------------------------------------------------------------------------


def read_csv(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    return pd.read_csv(path)

#------------------------------------------

def write_csv(dataframe, file_path):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)

#------------------------------------------

def statistics(dataframe):
    """Return descriptive statistics for an engineering dataset."""
    if dataframe.empty:
        raise ValueError("Dataset cannot be empty.")
    return dataframe.describe()

#------------------------------------------

def filter_data(dataframe, column, minimum):
    if column not in dataframe.columns:
        raise KeyError(f"Column not found: {column}")
    return dataframe[dataframe[column] >= minimum]

#------------------------------------------

def export_processed_data(dataframe, file_path):
    """Export processed engineering data to a CSV file."""
    if dataframe.empty:
        raise ValueError("Cannot export an empty dataset.")
    write_csv(dataframe, file_path)


# state variable---------------------------
current_df = None

def handle_load_csv():
    global current_df
    path_input = input(f"Enter file path (Default: datasets/flight_test.csv): ")
    file_path = path_input if path_input else "datasets/flight_test.csv"
    try:
        current_df = read_csv(file_path)
        print(f"\n✅ Dataset is successfully loaded! Total rows: {len(current_df)}")
    except Exception as e:
        print(f"Loading error: {e}")

#-------------------------------------------------------------------

def handle_show_stats():
    global current_df
    if current_df is None:
        print(f"\n⚠️ Firstly load any CSV file!  (Option:1)")
        return
    else:
        print("\n--- DATASET STATISTICS ---")
        print(f"{statistics(current_df)}")

#-------------------------------------------------------------------

def handle_filter_data():
    global current_df
    if current_df is None:
        print("\n⚠️ Firstly load any CSV file! (Option:1)")
        return
        
    try:
        column_names = current_df.columns.tolist()
        col = input(f"Select any one column {column_names}: ").strip()
        
        user_input = input("Write the minimum value as a filter: ").strip()
        val = float(user_input)
        _validate_number(val, "value")
        
        current_df = filter_data(current_df, col, val)
        print(f"Total filtered rows: {len(current_df)}")
        
        # Fixed Y/N input loop
        while True:
            show_filtered_data = input("\nDo you also want to see filtered data (Y/N): ").lower().strip()
            if show_filtered_data.startswith('y'):
                print(f"----- Rows with minimum value {val} of {col} -----")
                print(current_df)
                break
            elif show_filtered_data.startswith('n'):
                print("No problem.")
                break
            else:
                print("Please choose only Y/N.")

    
    except ValueError:
        print("\n❌ Invalid input! Please enter a valid number.")
    except Exception as e:
        print(f"\n❌ Filtering problem: {e}")
    
#-------------------------------------------------------------------

def handle_export_data():
    global current_df
    if current_df is None:
        print(f"\n⚠️ Firstly load any CSV file! (Option:1)")
        return
    print("Default path: reports/filtered_data.csv")
    path = input("Please write or paste the path where you want to export data (e.g. output.csv): ").strip()
    out_path = path if path else "reports/filtered_data.csv"
    try:
        export_processed_data(current_df, out_path)
        print(f"\n✅ Data is successfully exported: {out_path}")
    except Exception as e:
        print(f"\n❌ Export error: {e}")


# ==========================================
#            DICTIONARY MENU
# ==========================================

DATASET_MENU = {
    "1" : {"title" : "Load CSV Dataset", "func" : handle_load_csv},
    "2" : {"title" : "Show Statistics", "func" : handle_show_stats},
    "3": {"title": "Filter Data", "func": handle_filter_data},
    "4": {"title": "Export Processed Data", "func": handle_export_data},

}

def get_current_data():
    global current_df
    return current_df