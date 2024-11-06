import os
import yaml
import pandas as pd
import logging
from time import sleep
from metaapi_cloud_sdk import MetaApi
from datetime import datetime, timedelta
import asyncio
import pytz
import sys
from logging.handlers import RotatingFileHandler

# Setup logging
log_file = os.path.join('logs', 'candles.log')
os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

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

mt4_token = config['api_keys']['mt4_token']
mt4_account_id = config['api_keys']['mt4_account_id']
mt4_domain = config['api_keys']['domain']

# Ensure candles directory exists
candles_dir = os.path.join('data', 'candles')
os.makedirs(candles_dir, exist_ok=True)

# Function to clean and validate data before saving
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)
    return df

# Function to initialize MetaApi instance with retry logic
async def initialize_api():
    retries = 3
    for attempt in range(retries):
        try:
            api = MetaApi(mt4_token, {'domain': mt4_domain})
            logger.info("MetaApi instance initialized successfully.")
            return api
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - Failed to initialize MetaApi: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(5)  # Wait before retrying
            else:
                logger.error("Exceeded maximum retry attempts for MetaApi initialization.")
    return None

# Function to fetch historical data in chunks
async def fetch_data_in_chunks(account, symbol, start_time, end_time):
    chunk_size = timedelta(days=30)  # Fetch data in 30-day chunks
    current_start = start_time
    all_candles = []

    while current_start < end_time:
        current_end = min(current_start + chunk_size, end_time)
        logger.info(f'Fetching candles for {symbol} from {current_start} to {current_end}')
        candles = await account.get_historical_candles(symbol, '1h', current_start, current_end)
        all_candles.extend(candles)
        current_start = current_end

    return all_candles

# Function to fetch historical data up to today and keep fetching every day
async def retrieve_historical_candles():
    api = await initialize_api()
    if api is None:
        logger.error("MetaApi instance initialization failed, aborting historical candle retrieval.")
        return

    try:
        account = await api.metatrader_account_api.get_account(mt4_account_id)

        # Wait until account is deployed and connected to broker
        logger.info('Deploying account')
        if account.state != 'DEPLOYED':
            await account.deploy()
        else:
            logger.info('Account already deployed')
        logger.info('Waiting for API server to connect to broker (may take a couple of minutes)')
        if account.connection_status != 'CONNECTED':
            await account.wait_connected()

        while True:
            # Retrieve historical candles from 06-11-2022 to today
            end_time = datetime(2024, 11, 6, tzinfo=pytz.utc)
            start_time = datetime(2022, 11, 6, tzinfo=pytz.utc)
            logger.info(f'Downloading historical candles for {symbols} from {start_time} to {end_time}')

            for symbol in symbols:
                candles = await fetch_data_in_chunks(account, symbol, start_time, end_time)
                logger.info(f'Downloaded {len(candles)} historical candles for {symbol}')
                if candles:
                    df = pd.DataFrame(candles)
                    df['time'] = pd.to_datetime(df['time'])  # Ensure time is in datetime format
                    file_path = os.path.join(candles_dir, f'{symbol}_candles.csv')
                    
                    # If file exists, append new data; otherwise, create a new file
                    if os.path.exists(file_path):
                        existing_data = pd.read_csv(file_path)
                        existing_data['time'] = pd.to_datetime(existing_data['time'])  # Convert existing time column to datetime
                        combined_data = pd.concat([existing_data, df]).drop_duplicates()
                    else:
                        combined_data = df
                    
                    # Clean and validate data
                    combined_data = clean_data(combined_data)
                    
                    # Save to CSV
                    combined_data.to_csv(file_path, index=False)
                    logger.info(f'Saved {symbol} candles to {file_path}')
                    logger.info(f'First candle: {candles[0]}')
                    logger.info(f'Last candle: {candles[-1]}')

            logger.info("Historical data update complete. Sleeping until next update.")
            await asyncio.sleep(24 * 3600)  # Sleep for 24 hours before fetching again

    except Exception as err:
        logger.error(f"Error fetching historical candles: {err}")
    finally:
        # Only close if API was successfully initialized
        if api:
            logger.info("Closing MetaApi instance.")
            try:
                await api.close()
            except Exception as close_err:
                logger.error(f"Error closing MetaApi instance: {close_err}")
        else:
            logger.warning("MetaApi instance was not initialized, skipping close.")
        sys.exit()

# Run the historical data fetch loop
if __name__ == "__main__":
    try:
        asyncio.run(retrieve_historical_candles())
    except KeyboardInterrupt:
        logger.info("Historical data fetch loop interrupted by user.")

