import os
import yaml
import pandas as pd
import yfinance as yf
from typing import Dict
import logging
from time import sleep

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

symbols = config['assets']
resample_frequency = config.get('resample_frequency', '1h')
yf_period = '2y'
yf_interval = '1h'
yf_symbols = {
    'XAUUSD': 'GC=F',
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X'
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs('data', exist_ok=True)

def fetch_yfinance_data(symbol: str, retries: int = 3) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance with retry logic."""
    yf_symbol = yf_symbols.get(symbol, symbol)
    for attempt in range(retries):
        try:
            logger.info(f"Fetching {yf_period} of data for {symbol} from Yahoo Finance (attempt {attempt + 1})...")
            data = yf.download(yf_symbol, period=yf_period, interval=yf_interval)

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(1)
            data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

            logger.info(f"Successfully fetched data for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
            sleep(2)
    return pd.DataFrame()

def resample_data(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Resample data to specified frequency."""
    if not data.empty:
        data.index = pd.to_datetime(data.index)
        return data.resample(frequency).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    return pd.DataFrame()

def save_to_csv(data: pd.DataFrame, symbol: str):
    """Save Yahoo Finance data to CSV file."""
    if not data.empty:
        filepath = f'data/{symbol}_yfinance.csv'
        data.to_csv(filepath)
        logger.info(f"Saved data for {symbol} to {filepath}")

def main():
    """Main execution function."""
    logger.info("Starting historical data retrieval for all symbols...")
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        yf_data = fetch_yfinance_data(symbol)
        if not yf_data.empty:
            resampled_data = resample_data(yf_data, resample_frequency)
            save_to_csv(resampled_data, symbol)
        else:
            logger.warning(f"No data available for {symbol}")
    logger.info("Historical data retrieval completed.")

if __name__ == "__main__":
    main()
