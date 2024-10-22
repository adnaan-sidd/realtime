import pandas as pd
import numpy as np
import os
import yaml

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
        
        # Load CSV and rename columns to standard names
        df = pd.read_csv(file_path)
        
        # Assuming the CSV has columns: Type, Time, Open, High, Low, Close, Volume
        df.columns = ['Type', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert 'Time' column to datetime
        df['Time'] = pd.to_datetime(df['Time'], utc=True)
        
        # Set 'Time' as the index
        df.set_index('Time', inplace=True)
        
        # Create Bid and Ask columns (since we don't have separate Bid and Ask data)
        df['Bid_Open'] = df['Open']
        df['Bid_High'] = df['High']
        df['Bid_Low'] = df['Low']
        df['Bid_Close'] = df['Close']
        df['Ask_Open'] = df['Open']
        df['Ask_High'] = df['High']
        df['Ask_Low'] = df['Low']
        df['Ask_Close'] = df['Close']
        
        # Ensure numeric types for aggregation (handling non-numeric values)
        numeric_columns = ['Bid_Open', 'Bid_High', 'Bid_Low', 'Bid_Close', 
                           'Ask_Open', 'Ask_High', 'Ask_Low', 'Ask_Close', 'Volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        return df

    def resample_data(self, df):
        """Resample data to the configured frequency (e.g., 5 minutes, 1 hour)."""
        # Ensure that all necessary columns are numeric before resampling
        df = df[['Bid_Open', 'Bid_High', 'Bid_Low', 'Bid_Close', 
                 'Ask_Open', 'Ask_High', 'Ask_Low', 'Ask_Close', 'Volume']]
        
        # Convert resample_frequency to pandas offset string
        if self.resample_frequency[-1] == 'T':
            resample_freq = self.resample_frequency[:-1] + 'min'
        else:
            resample_freq = self.resample_frequency
        
        # Resample data using the OHLC method for bid and ask prices
        df_resampled = df.resample(resample_freq).agg({
            'Bid_Open': 'first',
            'Bid_High': 'max',
            'Bid_Low': 'min',
            'Bid_Close': 'last',
            'Ask_Open': 'first',
            'Ask_High': 'max',
            'Ask_Low': 'min',
            'Ask_Close': 'last',
            'Volume': 'sum'
        })
        
        return df_resampled

    def handle_missing_data(self, df):
        """Handle missing data by filling forward and dropping remaining NA values."""
        df.fillna(method='ffill', inplace=True)  # Fill missing data by forward filling
        df.dropna(inplace=True)                  # Drop any remaining rows with NA
        return df

    def add_features(self, df):
        """Add technical indicators and other derived features."""
        # Calculate percentage returns
        df['returns'] = df['Bid_Close'].pct_change()
        
        # Moving Average (SMA) for the close price
        df['SMA_20'] = df['Bid_Close'].rolling(window=20).mean()
        
        # Bollinger Bands
        df['BB_upper'] = df['SMA_20'] + 2 * df['Bid_Close'].rolling(window=20).std()
        df['BB_lower'] = df['SMA_20'] - 2 * df['Bid_Close'].rolling(window=20).std()

        # Relative Strength Index (RSI)
        delta = df['Bid_Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def normalize_data(self, df):
        """Normalize data (e.g., close price, returns) to [0, 1] for the LSTM model."""
        df_normalized = df.copy()
        df_normalized['close_normalized'] = (df['Bid_Close'] - df['Bid_Close'].min()) / (df['Bid_Close'].max() - df['Bid_Close'].min())
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

