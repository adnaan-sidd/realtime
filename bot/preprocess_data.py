import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import os
import pickle
import ast

def load_and_merge_data(symbol: str, sentiment_file: str) -> pd.DataFrame:
    """
    Load and merge market data with sentiment data.
    """
    try:
        # Load market data from yfinance
        yfinance_data = pd.read_csv(f'data/yfinance/{symbol}_yf.csv', index_col=0)
        yfinance_data.index = pd.to_datetime(yfinance_data.index, errors='coerce')

        # Load market data from candles
        candles_data = pd.read_csv(f'data/candles/{symbol}_candles.csv', index_col=0)
        candles_data.index = pd.to_datetime(candles_data.index, errors='coerce')

        # Combine both datasets
        market_data = pd.concat([yfinance_data, candles_data]).drop_duplicates().sort_index()

        # Load sentiment data
        sentiment_data = pd.read_csv(sentiment_file)
        if 'timestamp' not in sentiment_data.columns:
            raise ValueError("Sentiment data must contain a 'timestamp' column")

        sentiment_data['Date'] = pd.to_datetime(sentiment_data['timestamp'], errors='coerce')
        sentiment_data.set_index('Date', inplace=True)

        # Parse sentiment scores from dictionary string to separate columns
        sentiment_data['sentiment_scores'] = sentiment_data['sentiment_scores'].apply(ast.literal_eval)
        sentiment_data['positive'] = sentiment_data['sentiment_scores'].apply(lambda x: x['positive'])
        sentiment_data['negative'] = sentiment_data['sentiment_scores'].apply(lambda x: x['negative'])
        sentiment_data['neutral'] = sentiment_data['sentiment_scores'].apply(lambda x: x['neutral'])
        sentiment_data.drop(columns=['sentiment_scores'], inplace=True)

        # Merge data using an outer join to capture all records
        merged_data = pd.merge(market_data, sentiment_data, left_index=True, right_index=True, how='outer')

        print(f"Loaded and merged data for {symbol}. Shape: {merged_data.shape}")
        return merged_data

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        raise
    except Exception as e:
        print(f"Error during data loading and merging: {e}")
        raise

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for the dataset.
    """
    try:
        df = df.copy()

        # Calculate returns
        df['Returns'] = df['Close'].pct_change(fill_method=None)

        # Calculate moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()

        # Add volatility indicator
        df['Volatility'] = df['Close'].rolling(window=20).std()

        # Add momentum indicators
        df['Momentum'] = df['Close'] - df['Close'].shift(4)
        df['ROC'] = (df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12) * 100

        return df

    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        raise

def prepare_sequences(data: pd.DataFrame, sequence_length: int, target_column: str = 'Close', feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Prepare sequences for LSTM training.
    """
    try:
        if feature_columns is None:
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'positive', 'negative', 'neutral', 'Returns', 'RSI', 'Momentum', 'ROC']

        # Ensure columns are present
        missing_columns = [col for col in feature_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")

        scalers = {}
        scaled_data = pd.DataFrame()

        # Scale each feature
        for column in feature_columns:
            scaler = MinMaxScaler()
            scaled_data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1)).flatten()
            scalers[column] = scaler

        X, y = [], []

        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[feature_columns].iloc[i:(i + sequence_length)].values)
            y.append(scaled_data[target_column].iloc[i + sequence_length])

        return np.array(X), np.array(y), scalers

    except Exception as e:
        print(f"Error preparing sequences: {e}")
        raise

def preprocess_data(symbol: str, sentiment_file: str, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Main preprocessing function.
    """
    try:
        print(f"Starting preprocessing for {symbol}")

        # Load and merge data
        merged_data = load_and_merge_data(symbol, sentiment_file)

        # Calculate technical indicators
        data_with_indicators = calculate_technical_indicators(merged_data)

        # Fill NaNs with forward and backward fill
        data_filled = data_with_indicators.ffill().bfill()

        # Prepare sequences for LSTM
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'positive', 'negative', 'neutral', 'Returns', 'RSI', 'Momentum', 'ROC']
        X, y, scalers = prepare_sequences(data_filled, sequence_length=sequence_length, feature_columns=feature_columns)

        print(f"Preprocessed data shape for {symbol}: X: {X.shape}, y: {y.shape}")

        os.makedirs('data', exist_ok=True)

        # Save scalers
        with open(f'data/{symbol}_scalers.pkl', 'wb') as f:
            pickle.dump(scalers, f)

        return X, y, scalers

    except Exception as e:
        print(f"Error in preprocessing pipeline: {e}")
        raise

if __name__ == "__main__":
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "gold"]
    sentiment_file = "data/sentiment_data.csv"
    sequence_length = 60

    try:
        print("Starting preprocessing pipeline for multiple assets...")
        for symbol in symbols:
            X, y, scalers = preprocess_data(symbol, sentiment_file, sequence_length=sequence_length)

            # Save processed data
            np.save(f'data/{symbol}_X.npy', X)
            np.save(f'data/{symbol}_y.npy', y)
            print(f"Preprocessing completed successfully for {symbol}. Data saved in 'data/{symbol}_X.npy' and 'data/{symbol}_y.npy'")

    except Exception as e:
        print(f"Error in main execution: {e}")

