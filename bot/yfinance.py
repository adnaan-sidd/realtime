import os
import yaml
import pandas as pd
import yfinance as yf
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Load configuration
config_path = os.path.join('config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

symbols = config['assets']
yf_period = config['yf_period']
yf_interval = config['yf_interval']

# Mapping for Yahoo Finance symbols
yf_symbols = {
    'XAUUSD': 'GC=F',
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X'
}

# Set up logging with rotation
os.makedirs('logs', exist_ok=True)
log_file = 'logs/yfinance_data_collection.log'
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[handler, logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
data_dir = 'data/yfinance'
os.makedirs(data_dir, exist_ok=True)

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

            # Convert timezone-aware index to UTC
            data.index = data.index.tz_localize('UTC') if data.index.tz is None else data.index.tz_convert('UTC')

            logger.info(f"Successfully fetched data for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
    return pd.DataFrame()

def save_yfinance_data(data: pd.DataFrame, symbol: str):
    """Save Yahoo Finance data to CSV."""
    if not data.empty:
        filename = f'{symbol}_yf.csv'
        filepath = os.path.join(data_dir, filename)
        
        # If file exists, append new data; otherwise, create a new file
        if os.path.exists(filepath):
            existing_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            combined_data = pd.concat([existing_data, data]).drop_duplicates().sort_index()
        else:
            combined_data = data
        
        combined_data.to_csv(filepath)
        logger.info(f"Saved Yahoo Finance data for {symbol} to {filepath}")

def main():
    logger.info("Starting Yahoo Finance data collection...")
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")

        # Fetch and save Yahoo Finance data
        yf_data = fetch_yfinance_data(symbol)
        save_yfinance_data(yf_data, symbol)

    logger.info("Completed Yahoo Finance data collection.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

