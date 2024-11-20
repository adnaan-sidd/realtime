import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MT5Credentials:
    """Store MT5 connection credentials"""
    login: int
    password: str
    server: str

class MT5Handler:
    TIMEFRAME_MAP: Dict[str, int] = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1
    }

    TIMEFRAME_MINUTES: Dict[str, int] = {
        'M1': 1,
        'M5': 5,
        'M15': 15,
        'M30': 30,
        'H1': 60,
        'H4': 240,
        'D1': 1440,
        'W1': 10080,
        'MN1': 43200
    }

    def __init__(self, credentials: MT5Credentials):
        self.credentials = credentials
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def initialize(self) -> bool:
        """Initialize MT5 connection with error handling"""
        if self._initialized:
            return True

        success = mt5.initialize(
            login=self.credentials.login,
            password=self.credentials.password,
            server=self.credentials.server
        )

        if not success:
            error_code = mt5.last_error()
            logger.error(f"Failed to initialize MT5: {error_code}")
            return False

        self._initialized = True
        logger.info("MT5 connection initialized successfully")
        return True

    def shutdown(self) -> None:
        """Safely shutdown MT5 connection"""
        if self._initialized:
            mt5.shutdown()
            self._initialized = False
            logger.info("MT5 connection closed")

    @classmethod
    def get_mt5_timeframe(cls, timeframe: str) -> Optional[int]:
        """Convert string timeframe to MT5 timeframe constant"""
        return cls.TIMEFRAME_MAP.get(timeframe)

class DataManager:
    def __init__(self, base_dir: str = 'data'):
        self.base_dir = base_dir
        self.candles_dir = os.path.join(base_dir, 'candles')
        os.makedirs(self.candles_dir, exist_ok=True)

    def get_file_path(self, symbol: str, timeframe: str) -> str:
        return os.path.join(self.candles_dir, f"{symbol}_{timeframe}.csv")

    def get_last_candle_time(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the timestamp of the last candle in the existing file"""
        file_path = self.get_file_path(symbol, timeframe)
        if not os.path.exists(file_path):
            return None
        
        try:
            df = pd.read_csv(file_path)
            if len(df) == 0:
                return None
            return pd.to_datetime(df['time'].iloc[-1])
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    def update_data(self, symbol: str, timeframe: str, new_data: pd.DataFrame) -> bool:
        """Update existing file with new data"""
        file_path = self.get_file_path(symbol, timeframe)
        try:
            if os.path.exists(file_path):
                existing_data = pd.read_csv(file_path)
                existing_data['time'] = pd.to_datetime(existing_data['time'])
                
                # Combine existing and new data, remove duplicates
                combined_data = pd.concat([existing_data, new_data])
                combined_data = combined_data.drop_duplicates(subset=['time'], keep='last')
                combined_data = combined_data.sort_values('time')
                
                # Save updated data
                combined_data.to_csv(file_path, index=False)
                logger.info(f"Updated {symbol} {timeframe} with {len(new_data)} new candles")
            else:
                new_data.to_csv(file_path, index=False)
                logger.info(f"Created new file for {symbol} {timeframe} with {len(new_data)} candles")
            return True
        except Exception as e:
            logger.error(f"Error updating data for {symbol} {timeframe}: {e}")
            return False

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
def fetch_data_range(
    handler: MT5Handler,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """Fetch data for a specific date range with retry mechanism"""
    try:
        mt5_timeframe = handler.get_mt5_timeframe(timeframe)
        if mt5_timeframe is None:
            logger.error(f"Invalid timeframe: {timeframe}")
            return None

        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        if rates is None:
            logger.error(f"Failed to fetch data for {symbol} {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['download_time'] = datetime.now()
        
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
        return None

def update_symbol_data(
    handler: MT5Handler,
    data_manager: DataManager,
    symbol: str,
    timeframe: str,
    lookback_days: int = 730  # 2 years default
) -> bool:
    """Update data for a single symbol and timeframe"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        logger.info(f"Fetching data for {symbol} {timeframe} from {start_date} to {end_date}")

        df = fetch_data_range(handler, symbol, timeframe, start_date, end_date)
        if df is None or len(df) == 0:
            logger.info(f"No new data available for {symbol} {timeframe}")
            return True  # Not an error condition

        return data_manager.update_data(symbol, timeframe, df)

    except Exception as e:
        logger.error(f"Error updating {symbol} {timeframe}: {e}")
        return False

def update_all_symbols(
    symbols: List[str],
    timeframes: List[str],
    credentials: MT5Credentials,
    max_workers: int = 5
) -> None:
    """Update data for all symbols and timeframes once"""
    data_manager = DataManager()

    with MT5Handler(credentials) as handler:
        if not handler._initialized:
            logger.error("MT5 initialization failed")
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    update_symbol_data, handler, data_manager, symbol, timeframe
                ): (symbol, timeframe)
                for symbol in symbols
                for timeframe in timeframes
            }

            for future in as_completed(futures):
                symbol, timeframe = futures[future]
                try:
                    success = future.result()
                    if not success:
                        logger.warning(f"Failed to update {symbol} {timeframe}")
                except Exception as e:
                    logger.error(f"Task failed for {symbol} {timeframe}: {e}")

if __name__ == "__main__":
    # Load configuration from file
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    symbols = config['assets']
    timeframes = config['data_config']['timeframe']
    
    credentials = MT5Credentials(
        login=int(config['mt5_credentials']['account_number']),
        password=config['mt5_credentials']['password'],
        server=config['mt5_credentials']['server']
    )

    # Update all symbols and timeframes once
    update_all_symbols(
        symbols=symbols,
        timeframes=timeframes,  # Ensure timeframes is a list
        credentials=credentials
    )