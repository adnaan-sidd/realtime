import os
import yaml
import pandas as pd
import yfinance as yf
from typing import Dict
import logging
from time import sleep
from metaapi_cloud_sdk import MetaApi
from datetime import datetime, timedelta
import asyncio
import pytz
import sys
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

mt4_token = config['api_keys']['mt4_token']
mt4_account_id = config['api_keys']['mt4_account_id']
mt4_domain = config['api_keys']['domain']

# Set up logging with rotation
os.makedirs('logs', exist_ok=True)
log_file = 'logs/data_collection.log'
handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[handler, logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Create single data directory with source-specific subdirectories
os.makedirs('data', exist_ok=True)
os.makedirs('data/yfinance', exist_ok=True)
os.makedirs('data/metaapi', exist_ok=True)

async def fetch_yfinance_data(symbol: str, retries: int = 3) -> pd.DataFrame:
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
            sleep(2)
    return pd.DataFrame()

async def fetch_metaapi_ticks(symbol: str, start_time: datetime = None) -> pd.DataFrame:
    """Fetch historical ticks from MetaAPI with exponential backoff retries."""
    api = MetaApi(mt4_token, {'domain': mt4_domain})
    retries = 0
    max_retries = 5
    backoff_factor = 2

    while retries < max_retries:
        try:
            account = await api.metatrader_account_api.get_account(mt4_account_id)
            if account.state != 'DEPLOYED':
                await account.deploy()
            if account.connection_status != 'CONNECTED':
                await account.wait_connected()

            if start_time is None:
                start_time = datetime.now(pytz.UTC) - timedelta(days=730)  # 2 years of data
            
            logger.info(f'Downloading ticks for {symbol} starting from {start_time}')
            started_at = datetime.now().timestamp()
            offset = 0
            data = []
            
            while True:
                try:
                    ticks = await account.get_historical_ticks(symbol, start_time, offset)
                    if not ticks or len(ticks) == 0:
                        break
                    
                    logger.info(f'Downloaded {len(ticks)} historical ticks for {symbol}')
                    data.extend(ticks)
                    
                    # Save data in chunks to avoid memory issues
                    if len(data) >= 100000:  # Save every 100K ticks
                        await save_ticks_chunk(symbol, data)
                        data = []  # Clear the data list after saving
                    
                    start_time = ticks[-1]['time']
                    offset = 0
                    while offset < len(ticks) and ticks[-1 - offset]['time'].timestamp() == start_time.timestamp():
                        offset += 1
                    
                    logger.info(f'Last tick time is {start_time}, offset is {offset}')
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error fetching ticks: {e}")
                    await asyncio.sleep(5)
                    continue
                
            logger.info(f'Took {(datetime.now().timestamp() - started_at) * 1000}ms')
            
            # Save any remaining ticks
            if data:
                await save_ticks_chunk(symbol, data)
            
            return pd.DataFrame(data) if data else pd.DataFrame()
        
        except Exception as err:
            logger.error(api.format_error(err))
            retries += 1
            if retries < max_retries:
                backoff_time = backoff_factor ** retries
                logger.info(f"Retrying fetch_metaapi_ticks for {symbol} in {backoff_time} seconds (attempt {retries}/{max_retries})")
                await asyncio.sleep(backoff_time)
            else:
                logger.error(f"Failed to fetch MetaAPI ticks for {symbol} after {max_retries} attempts.")
                return pd.DataFrame()

async def save_ticks_chunk(symbol: str, ticks: list):
    """Save a chunk of ticks data to CSV."""
    try:
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{symbol}_mt4_{timestamp}.csv'
        filepath = os.path.join('data', 'metaapi', filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(ticks)} ticks to {filepath}")
    except Exception as e:
        logger.error(f"Error saving ticks chunk: {e}")

def save_yfinance_data(data: pd.DataFrame, symbol: str):
    """Save Yahoo Finance data to CSV."""
    if not data.empty:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{symbol}_yf_{timestamp}.csv'
        filepath = os.path.join('data', 'yfinance', filename)
        data.to_csv(filepath)
        logger.info(f"Saved Yahoo Finance data for {symbol} to {filepath}")

async def collect_data_continuously():
    """Continuously collect data with error handling and retries."""
    while True:
        try:
            logger.info("Starting new data collection cycle...")
            for symbol in symbols:
                logger.info(f"Processing {symbol}...")
                
                # Fetch and save Yahoo Finance data
                yf_data = await fetch_yfinance_data(symbol)
                save_yfinance_data(yf_data, symbol)
                
                # Fetch and save MetaAPI tick data
                metaapi_data = await fetch_metaapi_ticks(symbol)
                if not metaapi_data.empty:
                    save_ticks_chunk(symbol, metaapi_data.to_dict('records'))
                else:
                    logger.info(f"No MetaAPI tick data available for {symbol}")
                
                # Add a delay between symbols to avoid rate limiting
                await asyncio.sleep(5)
            
            # Wait for 1 hour before starting the next cycle
            logger.info("Completed data collection cycle. Waiting for 1 hour...")
            await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Error in data collection cycle: {e}")
            logger.info("Waiting 5 minutes before retry...")
            await asyncio.sleep(300)

if __name__ == "__main__":
    try:
        logger.info("Starting continuous data collection...")
        asyncio.run(collect_data_continuously())
    except KeyboardInterrupt:
        logger.info("Data collection stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
