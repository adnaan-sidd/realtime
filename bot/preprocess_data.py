import pandas as pd
from datetime import datetime
from indicators.indicators import TechnicalIndicators
from strategies.moving_average import MovingAverageStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_bands import BollingerBandsStrategy

# Load sentiment data from CSV
def load_sentiment_data(file_path='data/sentiment_data.csv'):
    sentiment_df = pd.read_csv(file_path, parse_dates=['timestamp'])
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], utc=True)
    return sentiment_df

# Load historical data
def load_historical_data(file_path):
    historical_df = pd.read_csv(file_path, parse_dates=['Time'])
    historical_df.rename(columns={'Time': 'timestamp'}, inplace=True)
    return historical_df

# Merge sentiment and historical data
def merge_data(sentiment_df, historical_df):
    merged_df = pd.merge(historical_df, sentiment_df, on='timestamp', how='left')
    merged_df.fillna(0, inplace=True)  # Fill NaN values with 0
    return merged_df

# Generate signals using all strategies
def generate_signals(df):
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    
    # Moving Average Strategy
    ma_strategy = MovingAverageStrategy()
    df = ma_strategy.generate_signals(df)
    
    # RSI Strategy
    rsi_strategy = RSIStrategy()
    df = rsi_strategy.generate_signals(df)
    
    # Bollinger Bands Strategy
    bb_strategy = BollingerBandsStrategy()
    df = bb_strategy.generate_signals(df)
    
    return df

# Main preprocessing function
def preprocess_data():
    sentiment_df = load_sentiment_data()
    assets = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY']
    processed_data = {}

    for asset in assets:
        historical_df = load_historical_data(f'data/{asset}.csv')
        merged_df = merge_data(sentiment_df, historical_df)
        final_df = generate_signals(merged_df)
        processed_data[asset] = final_df

    return processed_data

if __name__ == "__main__":
    processed_data = preprocess_data()
    for asset, df in processed_data.items():
        df.to_csv(f'data/processed_{asset}.csv', index=False)
        print(f"Processed data for {asset} saved to 'data/processed_{asset}.csv'")
