import yaml
import logging
import asyncio
import os
from typing import Dict, List, Optional
from datetime import datetime
from news.news_scraper import NewsScraperEnhanced
from candles import continuous_data_update
from yfinance import continuous_data_update as yfinance_update
from preprocess_data import main as preprocess_main
from models.lstm_model import train_model


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
                yaml.dump({'mt5_credentials': {}, 'assets': [], 'symbol_mapping': {}}, file)
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
        required_fields = ['mt5_credentials', 'assets', 'symbol_mapping']
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
    async def news_scraper_loop(self) -> None:
        """Run news scraping every hour."""
        news_scraper = NewsScraperEnhanced()
        while self.running:
            await asyncio.get_event_loop().run_in_executor(
                None, news_scraper.fetch_news
            )
            self.logger.info("News scraping completed")
            await asyncio.sleep(3600)  # Wait for 1 hour

    @async_error_handler
    async def candle_data_loop(self) -> None:
        """Run candle data updates every minute."""
        timeframes = ['M15', 'H1', 'H4', 'D1']
        credentials = self.get_mt5_credentials()
        while self.running:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: continuous_data_update(
                    symbols=self.config['assets'],
                    timeframes=timeframes,
                    credentials=credentials
                )
            )
            await asyncio.sleep(60)  # Wait for 1 minute

    @async_error_handler
    async def yfinance_data_loop(self) -> None:
        """Run YFinance updates every 360 seconds."""
        while self.running:
            yfinance_assets = [self.config['symbol_mapping'].get(asset, asset) for asset in self.config['assets']]
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: yfinance_update(yfinance_assets)
            )
            await asyncio.sleep(360)  # Wait for 360 seconds

    @async_error_handler
    async def analysis_loop(self) -> None:
        """Run preprocessing and model training every hour."""
        while self.running:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.run_preprocessing
            )

            predictions = await asyncio.get_event_loop().run_in_executor(
                None,
                self.train_models
            )
            self.logger.info(f"Completed hourly analysis cycle. Predictions: {predictions}")
            await asyncio.sleep(3600)  # Wait for 1 hour

    async def run_async(self) -> None:
        """Main asynchronous execution method."""
        try:
            self.load_config()
            self.running = True
            self.logger.info("Starting AsyncTradingBot...")

            # Create tasks for different update frequencies
            self.tasks = [
                asyncio.create_task(self.news_scraper_loop()),
                asyncio.create_task(self.candle_data_loop()),
                asyncio.create_task(self.yfinance_data_loop()),
                asyncio.create_task(self.analysis_loop())
            ]

            # Wait for all tasks to complete
            await asyncio.gather(*self.tasks)
        except Exception as e:
            self.logger.error("Trading bot execution failed", exc_info=True)
            self.running = False
            raise
        finally:
            self.running = False

    async def shutdown(self) -> None:
        """Stop all running loops and tasks."""
        self.logger.info("Shutting down AsyncTradingBot...")
        self.running = False
        await self.stop_tasks()

    async def stop_tasks(self) -> None:
        """Cancel all running tasks gracefully."""
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.logger.info("All tasks stopped.")

    def get_mt5_credentials(self) -> MT5Credentials:
        """Get MT5 credentials from config."""
        try:
            return MT5Credentials(
                login=int(self.config['mt5_credentials']['account_number']),
                password=self.config['mt5_credentials']['password'],
                server=self.config['mt5_credentials']['server']
            )
        except (KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Error processing MT5 credentials: {e}")
            raise ConfigurationError("Invalid MT5 credentials configuration")

    @async_error_handler
    def run_news_scraper(self) -> List[Dict]:
        """Execute news scraping operation."""
        news_scraper = NewsScraperEnhanced()
        return news_scraper.fetch_news()

    @async_error_handler
    def run_preprocessing(self) -> None:
        """Execute data preprocessing pipeline."""
        preprocess_main()

    @async_error_handler
    def train_models(self) -> Dict[str, Optional[float]]:
        """Train LSTM models for all symbols."""
        predictions = {}
        for symbol in self.config['assets']:
            prediction = train_model(symbol)
            predictions[symbol] = prediction
            if prediction is not None:
                self.logger.info(f"Latest prediction for {symbol}: {prediction}")
            else:
                self.logger.warning(f"Could not get prediction for {symbol}")
        return predictions


async def main():
    """Entry point for the async trading bot."""
    bot = AsyncTradingBot('config/config.yaml')
    try:
        await bot.run_async()
    except KeyboardInterrupt:
        bot.logger.info("Received shutdown signal, stopping bot...")
        await bot.shutdown()
    except Exception as e:
        bot.logger.error(f"Trading bot execution failed: {e}")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())