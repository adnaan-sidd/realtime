import yaml
import logging
import asyncio
import os
from datetime import datetime
from news.news_scraper import NewsScraperEnhanced
from candles import continuous_data_update  # Fixed import path
from yfinance import continuous_data_update as yfinance_update
from preprocess_data import main as preprocess_main
from models.lstm_model import train_model
from backtest import Backtest
from mt5_trader import MT5Trader

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class MT5Credentials:
    """Class to hold MT5 credentials with proper attribute access."""
    def __init__(self, login: int, password: str, server: str):
        self.login = login
        self.password = password
        self.server = server

def async_error_handler(func):
    """Decorator for handling errors in asynchronous functions."""
    async def wrapper(*args, **kwargs):
        operation_name = func.__name__
        try:
            await func(*args, **kwargs)
            args[0].logger.info(f"{operation_name} completed successfully.")
        except Exception as e:
            args[0].logger.error(f"Error in {operation_name}: {e}")
            await asyncio.sleep(60)  # Wait before retry
    return wrapper

class AsyncTradingBot:
    """Asynchronous trading bot class that orchestrates all operations with different update frequencies."""

    def __init__(self, config_path: str):
        """Initialize the trading bot with configuration."""
        self.config_path = config_path
        self.config = None
        self.logger = self._setup_logging()
        self.running = False
        self.tasks = []

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Configure and return logger instance."""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
        return logging.getLogger(__name__)

    def load_config(self) -> None:
        """Load and validate YAML configuration."""
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as file:
                yaml.dump({'mt5_credentials': {}, 'assets': [], 'symbol_mapping': {}, 'api_keys': {}}, file)
            self.logger.warning(f"Default config created. Please update it at {self.config_path}")
            raise ConfigurationError(f"Default config created. Update it at {self.config_path}")

        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            self.logger.debug(f"Loaded configuration: {self.config}")
            self._validate_config()
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_fields = ['mt5_credentials', 'assets', 'symbol_mapping', 'api_keys']
        mt5_required_fields = ['account_number', 'password', 'server']

        missing_fields = [field for field in required_fields if field not in self.config]
        if missing_fields:
            self.logger.error(f"Missing required fields in config: {missing_fields}")
            raise ConfigurationError(f"Missing required fields in config: {missing_fields}")

        missing_mt5_fields = [field for field in mt5_required_fields if field not in self.config['mt5_credentials']]
        if missing_mt5_fields:
            self.logger.error(f"Missing required MT5 credentials: {missing_mt5_fields}")
            raise ConfigurationError(f"Missing required MT5 credentials: {missing_mt5_fields}")

        if not isinstance(self.config['assets'], list) or not self.config['assets']:
            self.logger.error("Assets list is empty or invalid")
            raise ConfigurationError("Assets list is empty or invalid")

    @async_error_handler
    async def fetch_news(self) -> None:
        """Fetch news articles."""
        # Accessing the API key from the nested structure
        bing_api_key = self.config.get('api_keys', {}).get('bing_api_key')
        if not bing_api_key:
            self.logger.error("Bing API key is not configured.")
            raise ConfigurationError("Bing API key is missing in the configuration.")
        
        scraper = NewsScraperEnhanced(api_key=bing_api_key)  # Use the loaded API key
        scraper.fetch_news()  # Correct method call
        self.logger.info("News articles fetched.")

    @async_error_handler
    async def update_candles(self) -> None:
        """Update candle data."""
        continuous_data_update()
        self.logger.info("Candle data updated.")

    @async_error_handler
    async def update_yfinance(self) -> None:
        """Update yfinance data."""
        yfinance_update()
        self.logger.info("YFinance data updated.")

    @async_error_handler
    async def preprocess_data(self) -> None:
        """Preprocess data."""
        preprocess_main()
        self.logger.info("Data preprocessed.")

    @async_error_handler
    async def train_model(self) -> None:
        """Train the model."""
        train_model()
        self.logger.info("Model trained.")

    @async_error_handler
    async def run_backtest(self) -> None:
        """Run backtest."""
        backtest = Backtest()
        backtest.run()  # Assuming you have a run method in Backtest
        self.logger.info("Backtest completed.")

    @async_error_handler
    async def execute_trades(self) -> None:
        """Execute trades."""
        trader = MT5Trader(self.config_path)
        trader.trade_on_predictions()
        trader.shutdown_mt5()
        self.logger.info("Trades executed.")

    async def run_tasks(self) -> None:
        """Run all tasks in sequence with a delay."""
        while self.running:
            await self.fetch_news()
            await asyncio.sleep(3600)  # Sleep for 1 hour

            await self.update_candles()
            await asyncio.sleep(3600)  # Sleep for 1 hour

            await self.update_yfinance()
            await asyncio.sleep(3600)  # Sleep for 1 hour

            await self.preprocess_data()
            await asyncio.sleep(3600)  # Sleep for 1 hour

            await self.train_model()
            await asyncio.sleep(3600)  # Sleep for 1 hour

            await self.run_backtest()
            await asyncio.sleep(3600)  # Sleep for 1 hour

            await self.execute_trades()
            await asyncio.sleep(3600)  # Sleep for 1 hour

    def start(self) -> None:
        """Start the trading bot."""
        self.load_config()
        self.running = True
        asyncio.run(self.run_tasks())

    def stop(self) -> None:
        """Stop the trading bot."""
        self.running = False

if __name__ == "__main__":
    bot = AsyncTradingBot('config/config.yaml')
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
        bot.logger.info("Trading bot stopped.")