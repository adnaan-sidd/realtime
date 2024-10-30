from metaapi_cloud_sdk import MetaStats, MetaApi
import asyncio
import logging
import numpy as np
from models.lstm_model import make_predictions  # Adjust import based on your structure

class MetaApiTrader:
    def __init__(self, config):
        self.token = config['api_keys']['mt4_token']
        self.account_id = config['api_keys']['mt4_account_id']
        self.api = MetaApi(self.token)
        self.meta_stats = MetaStats(self.token)
        self.account = None
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']  # Define your trading symbols

    async def connect_account(self):
        try:
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)

            # Deploy account if not already deployed
            if self.account.state != 'DEPLOYED':
                await self.account.deploy()
                print("Account is being deployed...")
            else:
                print("Account is already deployed.")

            # Wait for the account to connect
            await self.account.wait_connected()
            print("Account is connected to the broker.")
        except Exception as err:
            logging.error(f"Error connecting to account: {err}")

    async def fetch_metrics(self):
        try:
            metrics = await self.meta_stats.get_metrics(self.account_id)
            print("Account Metrics:", metrics)
            return metrics
        except Exception as err:
            logging.error(f"Error fetching metrics: {err}")

    async def fetch_recent_trades(self, start_date='0000-01-01 00:00:00.000', end_date='9999-01-01 00:00:00.000'):
        try:
            trades = await self.meta_stats.get_account_trades(self.account_id, start_date, end_date)
            print("Recent Trades:", trades[-5:])  # Print last 5 trades
            return trades
        except Exception as err:
            logging.error(f"Error fetching trades: {err}")

    async def execute_trade(self, symbol, trade_type, amount):
        try:
            # Define order parameters
            order = {
                'symbol': symbol,
                'action': trade_type,  # "BUY" or "SELL"
                'amount': amount,
                'type': 'market',  # Market order
            }
            # Place the trade
            result = await self.api.trading_api.place_order(self.account_id, order)
            print(f"Trade executed: {result}")
        except Exception as err:
            logging.error(f"Error placing trade: {err}")

    async def trade_based_on_predictions(self):
        for symbol in self.symbols:
            # Get predictions from the LSTM model
            predictions = make_predictions(symbol)
            if predictions is not None:
                latest_prediction = predictions[-1]  # Get the latest prediction
                # Define trade logic based on prediction
                trade_type = "BUY" if latest_prediction > 0 else "SELL"  # Example logic
                amount = 0.1  # Define your amount

                # Execute the trade
                await self.execute_trade(symbol, trade_type, amount)

    async def run(self):
        await self.connect_account()
        await self.fetch_metrics()
        await self.fetch_recent_trades()
        await self.trade_based_on_predictions()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Example usage
if __name__ == "__main__":
    # Load your config here
    import yaml
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    trader = MetaApiTrader(config)
    asyncio.run(trader.run())
