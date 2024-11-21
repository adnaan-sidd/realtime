import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import yaml
import logging
import asyncio
import os
from datetime import datetime
from news.news_scraper import NewsScraperEnhanced
from candles import continuous_data_update as candles_update
from yfinance import continuous_data_update as yfinance_update
from preprocess_data import main as preprocess_main
from models.lstm_model import train_model
from backtest import Backtest
from mt5_trader import MT5Trader


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class AsyncTradingBot:
    """Asynchronous trading bot class that orchestrates all operations."""

    def __init__(self, config_path: str):
        """Initialize the trading bot with configuration."""
        self.config_path = config_path
        self.config = None
        self.logger = self._setup_logging()

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
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise ConfigurationError("Configuration file not found.")

        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            if not self.config.get('assets'):
                raise ConfigurationError("Assets not specified in the configuration file.")
            self.logger.info("Configuration loaded successfully.")
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration: {e}")
            raise ConfigurationError("Error parsing YAML configuration.")

    async def fetch_news(self) -> None:
        """Fetch news articles."""
        try:
            scraper = NewsScraperEnhanced(config_path="config/config.yaml")
            scraper.fetch_news()
            self.logger.info("News articles fetched.")
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            raise

    async def update_candles(self) -> None:
        """Update candle data."""
        try:
            candles_update()
            self.logger.info("Candle data updated.")
        except Exception as e:
            self.logger.error(f"Error updating candle data: {e}")
            raise

    async def update_yfinance(self) -> None:
        """Update Yahoo Finance data."""
        try:
            # Directly specify the correct symbols
            assets = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F"]
            self.logger.info(f"Updating Yahoo Finance data for assets: {assets}")
            yfinance_update(assets=assets)
            self.logger.info("Yahoo Finance data updated.")
        except Exception as e:
            self.logger.error(f"Error updating Yahoo Finance data: {e}")
            raise

    async def preprocess_data(self) -> None:
        """Preprocess data."""
        try:
            preprocess_main()
            self.logger.info("Data preprocessing completed.")
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {e}")
            raise

    async def train_models(self) -> None:
        """Train models for each asset."""
        try:
            for asset in self.config['assets']:
                self.logger.info(f"Training model for {asset}...")
                train_model(symbol=asset)
            self.logger.info("All models trained successfully.")
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            raise

    async def run_backtest(self) -> None:
        """Run backtest."""
        try:
            assets = self.config.get('assets', [])
            initial_balance = self.config.get('initial_balance', 10000)
            backtest = Backtest(assets, initial_balance)
            backtest.run_backtest()
            self.logger.info("Backtest completed.")
        except Exception as e:
            self.logger.error(f"Error during backtest: {e}")
            raise

    async def execute_trades(self) -> None:
        """Execute trades."""
        try:
            trader = MT5Trader(self.config_path)
            trader.trade_on_predictions()
            trader.shutdown_mt5()
            self.logger.info("Trades executed successfully.")
        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")
            raise

    async def run_sequence(self) -> None:
        """Run all tasks in sequence, then pause for 30 minutes."""
        while True:
            try:
                await self.fetch_news()
                await self.update_candles()
                await self.update_yfinance()
                await self.preprocess_data()
                await self.train_models()
                await self.run_backtest()
                await self.execute_trades()
                self.logger.info("All tasks completed. Waiting for 30 minutes...")
                await asyncio.sleep(1800)  # 30-minute delay
            except Exception as e:
                self.logger.error(f"Error during execution: {e}")
                self.logger.info("Retrying in 5 minutes...")
                await asyncio.sleep(300)  # Retry after 5 minutes

    def start(self) -> None:
        """Start the trading bot."""
        try:
            self.load_config()
            asyncio.run(self.run_sequence())
        except KeyboardInterrupt:
            self.logger.info("Trading bot stopped.")
        except ConfigurationError as e:
            self.logger.error(f"Configuration error: {e}")


if __name__ == "__main__":
    bot = AsyncTradingBot('config/config.yaml')
    bot.start()