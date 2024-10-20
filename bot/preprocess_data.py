# preprocess_data.py

import pandas as pd
import numpy as np
import os
import yaml  # <-- Added this import for YAML

class DataPreprocessor:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.assets = config['assets']
        self.resample_frequency = config['preprocessing']['resample_frequency']
        self.lookback_period = config['preprocessing']['lookback_period']
        self.data_folder = 'data'  # Folder where raw CSV files are stored
        self.processed_folder = 'data'  # Folder where processed CSV files will be stored

    def load_raw_data(self, asset):
        """Load raw historical data for a given asset."""
        file_path = os.path.join(self.data_folder, f"{asset}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No raw data found for {asset} at {file_path}")

        # Load the CSV and print the column names for inspection
        df = pd.read_csv(file_path)
        print(f"Columns in {asset}.csv: {df.columns}")

        # Automatically find the datetime column
        datetime_column = None
        for column in df.columns:
            if "date" in column.lower() or "time" in column.lower():
                datetime_column = column
                break

        if not datetime_column:
            raise ValueError(f"No datetime column found in {asset}.csv. Ensure the file contains a date or time column.")

        # Parse the detected datetime column and set it as the index
        df = pd.read_csv(file_path, parse_dates=[datetime_column], index_col=datetime_column)
        print(f"Using '{datetime_column}' as the datetime column for {asset}.csv")
        return df
    
    def resample_data(self, df):
        """Resample data to the configured frequency (e.g., 5 minutes, 1 hour)."""
        df_resampled = df.resample(self.resample_frequency).ohlc()
        df_resampled['volume'] = df['volume'].resample(self.resample_frequency).sum()
        return df_resampled

    def handle_missing_data(self, df):
        """Handle missing data by filling forward and dropping remaining NA values."""
        df.fillna(method='ffill', inplace=True)  # Fill missing data by forward filling
        df.dropna(inplace=True)                  # Drop any remaining rows with NA
        return df

    def add_features(self, df):
        """Add technical indicators and other derived features."""
        # Calculate percentage returns
        df['returns'] = df['close'].pct_change()
        
        # Moving Average (SMA) for the close price
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        
        # Bollinger Bands
        df['BB_upper'] = df['SMA_20'] + 2 * df['close'].rolling(window=20).std()
        df['BB_lower'] = df['SMA_20'] - 2 * df['close'].rolling(window=20).std()

        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def normalize_data(self, df):
        """Normalize data (e.g., close price, returns) to [0, 1] for the LSTM model."""
        df_normalized = df.copy()
        df_normalized['close_normalized'] = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())
        return df_normalized

    def preprocess_asset_data(self, asset):
        """Preprocess historical data for a given asset."""
        print(f"Preprocessing data for {asset}...")
        
        # Load and preprocess data
        df = self.load_raw_data(asset)
        df = self.resample_data(df)
        df = self.handle_missing_data(df)
        df = self.add_features(df)
        df = self.normalize_data(df)
        
        # Save the preprocessed data to a new CSV
        output_path = os.path.join(self.processed_folder, f"{asset}_processed.csv")
        df.to_csv(output_path)
        print(f"Processed data saved to {output_path}")

    def preprocess_all_assets(self):
        """Preprocess data for all assets defined in config."""
        for asset in self.assets:
            self.preprocess_asset_data(asset)

# Example usage:
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_all_assets()

