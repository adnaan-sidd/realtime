import os
import pandas as pd

def check_csv_columns_and_rows_in_directory(directory):
    """Check and print the column names and number of rows of all CSV files in a directory."""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(file_path)
                print(f"Columns in {filename}:")
                print(df.columns.tolist())
                print(f"Number of rows in {filename}: {len(df)}\n")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

# Example usage
directory = 'data'
check_csv_columns_and_rows_in_directory(directory)

