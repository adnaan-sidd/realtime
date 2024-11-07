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

def load_config():
    """Load and return configuration from yaml file"""
    try:
        config_path = os.path.join('config', 'config.yaml')
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate candlestick data"""
    df.drop_duplicates(inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)
    return df

async def initialize_api(config):
    """Initialize MetaAPI with retry logic"""
    retries = 3
    for attempt in range(retries):
        try:
            api = MetaApi(config['api_keys']['mt4_token'], {'domain': config['api_keys']['domain']})
            logger.info("MetaApi instance initialized successfully.")
            return api
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} - Failed to initialize MetaApi: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(5)
            else:
                logger.error("Exceeded maximum retry attempts for MetaApi initialization.")
    return None

async def fetch_data_in_chunks(account, symbol, start_time, end_time):
    """Fetch historical data in manageable chunks"""
    chunk_size = timedelta(days=30)
    current_start = start_time
    all_candles = []

    while current_start < end_time:
        current_end = min(current_start + chunk_size, end_time)
        logger.info(f'Fetching candles for {symbol} from {current_start} to {current_end}')
        try:
            candles = await account.get_historical_candles(symbol, '1h', current_start, current_end)
            if candles:
                all_candles.extend(candles)
            await asyncio.sleep(1)  # Prevent rate limiting
            current_start = current_end
        except Exception as e:
            logger.error(f"Error fetching chunk for {symbol}: {e}")
            await asyncio.sleep(5)

    return all_candles

async def fetch_historical_data(account, symbol, candles_dir):
    """Fetch 2 years of historical data for a symbol"""
    end_time = datetime.now(pytz.utc)
    start_time = end_time - timedelta(days=730)  # 2 years of data
    
    logger.info(f'Fetching 2 years of historical data for {symbol} from {start_time} to {end_time}')
    
    candles = await fetch_data_in_chunks(account, symbol, start_time, end_time)
    if candles:
        df = pd.DataFrame(candles)
        file_path = os.path.join(candles_dir, f'{symbol}_candles.csv')
        df = clean_data(df)
        df.to_csv(file_path, index=False)
        logger.info(f'Saved historical data for {symbol} to {file_path}')
    return bool(candles)

async def fetch_candles_data(symbols):
    """Fetch latest candle data for the given symbols"""
    config = load_config()
    api = await initialize_api(config)
    
    if api is None:
        logger.error("MetaApi instance initialization failed")
        return

    try:
        account = await api.metatrader_account_api.get_account(config['api_keys']['mt4_account_id'])

        # Deploy and connect account if needed
        if account.state != 'DEPLOYED':
            logger.info('Deploying account...')
            await account.deploy()
        
        if account.connection_status != 'CONNECTED':
            logger.info('Waiting for broker connection...')
            await account.wait_connected()

        candles_dir = os.path.join('data', 'candles')
        os.makedirs(candles_dir, exist_ok=True)

        # Check if historical data exists, if not fetch it first
        for symbol in symbols:
            file_path = os.path.join(candles_dir, f'{symbol}_candles.csv')
            if not os.path.exists(file_path):
                logger.info(f'No historical data found for {symbol}. Fetching 2 years of data first...')
                success = await fetch_historical_data(account, symbol, candles_dir)
                if not success:
                    logger.error(f'Failed to fetch historical data for {symbol}')
                    continue

        # Now fetch the latest data
        end_time = datetime.now(pytz.utc)
        start_time = end_time - timedelta(days=1)  # Get last 24 hours
        
        for symbol in symbols:
            try:
                logger.info(f'Fetching latest data for {symbol}')
                candles = await fetch_data_in_chunks(account, symbol, start_time, end_time)
                
                if candles:
                    file_path = os.path.join(candles_dir, f'{symbol}_candles.csv')
                    new_df = pd.DataFrame(candles)
                    
                    if os.path.exists(file_path):
                        existing_data = pd.read_csv(file_path)
                        existing_data['time'] = pd.to_datetime(existing_data['time'])
                        combined_data = pd.concat([existing_data, new_df])
                        combined_data = clean_data(combined_data)
                        combined_data.to_csv(file_path, index=False)
                        logger.info(f'Updated data for {symbol}')
                    else:
                        new_df = clean_data(new_df)
                        new_df.to_csv(file_path, index=False)
                        logger.info(f'Saved new data for {symbol}')

            except Exception as symbol_err:
                logger.error(f"Error processing {symbol}: {symbol_err}")

    except Exception as err:
        logger.error(f"Error in fetch_candles_data: {err}")
    finally:
        if api:
            try:
                await api.close()
            except Exception as close_err:
                logger.error(f"Error closing MetaApi connection: {close_err}")

if __name__ == "__main__":
    # When run directly, fetch historical data first
    async def init_historical_data():
        config = load_config()
        api = await initialize_api(config)
        
        if api:
            try:
                account = await api.metatrader_account_api.get_account(config['api_keys']['mt4_account_id'])
                
                if account.state != 'DEPLOYED':
                    await account.deploy()
                
                if account.connection_status != 'CONNECTED':
                    await account.wait_connected()
                
                candles_dir = os.path.join('data', 'candles')
                os.makedirs(candles_dir, exist_ok=True)
                
                for symbol in config['assets']:
                    await fetch_historical_data(account, symbol, candles_dir)
                
            finally:
                await api.close()
    
    try:
        asyncio.run(init_historical_data())
    except KeyboardInterrupt:
        logger.info("Historical data fetch interrupted by user.")
