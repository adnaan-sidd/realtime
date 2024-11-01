import pandas as pd
from models.lstm_model import make_predictions
from portfolio import PortfolioManager
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

class Backtest:
    def __init__(self, assets, initial_balance=10000):
        self.assets = assets
        self.portfolio = PortfolioManager(initial_balance)

    def simulate_trade(self, symbol, signal, price, volume):
        if signal == "Buy":
            self.portfolio.open_position(symbol=symbol, side="buy", volume=volume, price=price)
        elif signal == "Sell":
            self.portfolio.open_position(symbol=symbol, side="sell", volume=volume, price=price)

    def run_backtest_for_asset(self, symbol):
        data = pd.read_csv(f"data/{symbol}_yfinance.csv", index_col="Datetime", parse_dates=True)
        
        # Get predictions and handle cases where a single value might be returned
        predictions = make_predictions(symbol)
        
        # Ensure predictions is iterable and has at least two values
        if isinstance(predictions, (float, int)):
            logging.warning(f"Only one prediction received for {symbol}, skipping backtest.")
            return
        elif predictions is None or len(predictions) < 2:
            logging.warning(f"No valid predictions for {symbol}, skipping backtest.")
            return

        # Run backtest using predictions
        for idx in range(len(predictions) - 1):
            signal = "Buy" if predictions[idx + 1] > predictions[idx] else "Sell"
            price = data["Close"].iloc[idx + 1]
            self.simulate_trade(symbol, signal, price, volume=0.1)

        # Log final results for the asset
        logging.info(f"Backtest for {symbol} completed. Final balance: {self.portfolio.get_balance()}")
        logging.info(f"Open positions for {symbol}: {self.portfolio.positions}")

    def run_backtest(self):
        for symbol in self.assets:
            logging.info(f"\nStarting backtest for {symbol}...")
            self.run_backtest_for_asset(symbol)

if __name__ == "__main__":
    assets = config["assets"]
    backtest = Backtest(assets)
    backtest.run_backtest()
