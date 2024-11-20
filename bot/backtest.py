import pandas as pd
from models.lstm_model import make_predictions
from portfolio import PortfolioManager
import yaml
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

class Backtest:
    def __init__(self, assets, initial_balance=10000):
        self.assets = assets
        self.portfolio = PortfolioManager(initial_balance)

    def simulate_trade(self, symbol, signal, price, volume):
        try:
            if signal == "Buy":
                self.portfolio.open_position(symbol=symbol, side="buy", volume=volume, price=price)
            elif signal == "Sell":
                self.portfolio.open_position(symbol=symbol, side="sell", volume=volume, price=price)
            logger.info(f"Simulated {signal} trade for {symbol} at price {price} with volume {volume}")
        except Exception as e:
            logger.error(f"Error simulating trade for {symbol}: {e}")

    def run_backtest_for_asset(self, symbol):
        try:
            data_file = os.path.join("data", "yfinance", f"{symbol}_yf.csv")
            if symbol == "XAUUSD":
                data_file = os.path.join("data", "yfinance", "gold_yf.csv")
            
            data = pd.read_csv(data_file, parse_dates=True)
            data.set_index('ticker', inplace=True)  # Assuming 'ticker' is the unique identifier

            # Get predictions and handle cases where a single value might be returned
            predictions, duration = make_predictions(symbol)

            # Ensure predictions is iterable and has at least two values
            if isinstance(predictions, (float, int)):
                logger.warning(f"Only one prediction received for {symbol}, skipping backtest.")
                return
            elif predictions is None or len(predictions) < 2:
                logger.warning(f"No valid predictions for {symbol}, skipping backtest.")
                return

            # Run backtest using predictions
            for idx in range(len(predictions) - 1):
                if idx + 1 >= len(data):
                    logger.warning(f"Index {idx + 1} out of bounds for {symbol}, stopping backtest.")
                    break
                signal = "Buy" if predictions[idx + 1] > predictions[idx] else "Sell"
                price = data["close"].iloc[idx + 1]
                self.simulate_trade(symbol, signal, price, volume=0.1)

            # Log final results for the asset
            logger.info(f"Backtest for {symbol} completed. Final balance: {self.portfolio.get_balance()}")
            logger.info(f"Open positions for {symbol}: {self.portfolio.positions}")

        except FileNotFoundError as e:
            logger.error(f"Data file for {symbol} not found: {e}")
        except KeyError as e:
            logger.error(f"Column not found in data for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}")

    def run_backtest(self):
        for symbol in self.assets:
            logger.info(f"\nStarting backtest for {symbol}...")
            self.run_backtest_for_asset(symbol)

if __name__ == "__main__":
    assets = config["assets"]
    initial_balance = config.get("initial_balance", 10000)
    backtest = Backtest(assets, initial_balance)
    backtest.run_backtest()