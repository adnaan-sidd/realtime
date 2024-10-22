import pandas as pd
import numpy as np
import os
import yaml
import json
from datetime import datetime

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
        self.sentiment_file = 'sentiment.json'  # Path to sentiment file

    def load_raw_data(self, asset):
        """Load raw historical data for a given asset."""
        file_path = os.path.join(self.data_folder, f"{asset}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No raw data found for {asset} at {file_path}")
        
        df = pd.read_csv(file_path)
        df.columns = ['Type', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Time'] = pd.to_datetime(df['Time'], utc=True)  # Ensure time is UTC
        df.set_index('Time', inplace=True)

        # Add Bid/Ask columns to match expected structure
        df['Bid_Open'] = df['Open']
        df['Bid_High'] = df['High']
        df['Bid_Low'] = df['Low']
        df['Bid_Close'] = df['Close']
        df['Ask_Open'] = df['Open']
        df['Ask_High'] = df['High']
        df['Ask_Low'] = df['Low']
        df['Ask_Close'] = df['Close']
        
        return df

    def load_sentiment_data(self, asset):
        """Load sentiment data from sentiment.json and return DataFrame."""
        if not os.path.exists(self.sentiment_file):
            print(f"Sentiment file not found at {self.sentiment_file}")
            return pd.DataFrame()  # Return empty DataFrame if sentiment data is missing

        with open(self.sentiment_file, 'r') as f:
            sentiment_data = json.load(f)

        # Extract sentiment data for the current asset
        asset_sentiment = [entry for entry in sentiment_data if entry['asset'] == asset]
        sentiment_df = pd.DataFrame(asset_sentiment)

        if not sentiment_df.empty:
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
            # Convert to UTC timezone to match price_df
            sentiment_df['timestamp'] = sentiment_df['timestamp'].dt.tz_localize('UTC')

            sentiment_df.set_index('timestamp', inplace=True)

        return sentiment_df

    def merge_data(self, price_df, sentiment_df):
        """Merge price data and sentiment data based on timestamp."""
        if sentiment_df.empty:
            print("No sentiment data to merge.")
            price_df['sentiment'] = 0  # Default to neutral sentiment if no sentiment data is available
            return price_df

        # Ensure both dataframes are sorted by their index (timestamp) before merging
        price_df.sort_index(inplace=True)
        sentiment_df.sort_index(inplace=True)

        # Merge sentiment with price data using the nearest timestamp match
        merged_df = pd.merge_asof(price_df, sentiment_df[['sentiment']], left_index=True, right_index=True, direction='backward')
        return merged_df

    def resample_data(self, df):
        """Resample data to the configured frequency (e.g., 5 minutes, 1 hour)."""
        df_resampled = df.resample(self.resample_frequency).agg({
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
        """Preprocess historical data for a given asset and merge sentiment data."""
        print(f"Preprocessing data for {asset}...")

        # Load and preprocess price data
        df = self.load_raw_data(asset)
        df = self.resample_data(df)

        # Load sentiment data and merge with price data
        sentiment_df = self.load_sentiment_data(asset)
        df = self.merge_data(df, sentiment_df)

        # Add technical features (e.g., moving averages, RSI)
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

