from metaapi_cloud_sdk import MetaApi, MetaStats
import asyncio
import logging
import numpy as np
from models.lstm_model import make_predictions  # Adjust import based on your structure
import uuid
from datetime import datetime

class MetaApiTrader:
    def __init__(self, config):
        self.token = config['api_keys']['mt4_token']
        self.account_id = config['api_keys']['mt4_account_id']
        self.api = MetaApi(self.token)
        self.meta_stats = MetaStats(self.token)
        self.account = None
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']  # Define your trading symbols

        # Risk management parameters from config
        self.take_profit = config['risk_management'].get('take_profit', 0.02)  # 2% default
        self.stop_loss = config['risk_management'].get('stop_loss', 0.01)      # 1% default

    async def connect_account(self):
        """Connect to the MetaApi account."""
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
        """Fetch account metrics."""
        try:
            metrics = await self.meta_stats.get_metrics(self.account_id)
            print("Account Metrics:", metrics)
            return metrics
        except Exception as err:
            logging.error(f"Error fetching metrics: {err}")

    async def fetch_recent_trades(self, start_date='0000-01-01 00:00:00.000', end_date='9999-01-01 00:00:00.000'):
        """Fetch recent trades within a specified time range."""
        try:
            trades = await self.meta_stats.get_account_trades(self.account_id, start_date, end_date)
            print("Recent Trades:", trades[-5:])  # Print last 5 trades
            return trades
        except Exception as err:
            logging.error(f"Error fetching trades: {err}")

    async def execute_trade(self, symbol, trade_type, amount):
        """Execute a market order with risk management, checking market hours."""
        try:
            # Get an RPC connection to place the trade
            connection = self.account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()

            # Get and parse the server time
            server_time_data = await connection.get_server_time()
            server_time = datetime.fromisoformat(server_time_data['brokerTime'])

            # Check if the market is open for the symbol
            if not self.is_market_open(symbol, server_time):
                logging.warning(f"Market is closed for {symbol}. Trade not executed.")
                return

            # Generate a compliant clientId format
            client_id = f"TE_{symbol}_{uuid.uuid4().hex[:8]}"

            # Define take-profit and stop-loss prices based on market price
            price_data = await connection.get_symbol_price(symbol)
            open_price = price_data['ask'] if trade_type == "ORDER_TYPE_BUY" else price_data['bid']
            tp_price = open_price * (1 + self.take_profit) if trade_type == "ORDER_TYPE_BUY" else open_price * (1 - self.take_profit)
            sl_price = open_price * (1 - self.stop_loss) if trade_type == "ORDER_TYPE_BUY" else open_price * (1 + self.stop_loss)

            # Execute buy or sell based on trade_type with risk management
            if trade_type == "ORDER_TYPE_BUY":
                result = await connection.create_market_buy_order(
                    symbol=symbol,
                    volume=amount,
                    options={'comment': 'buy', 'clientId': client_id, 'takeProfit': tp_price, 'stopLoss': sl_price}
                )
            else:
                result = await connection.create_market_sell_order(
                    symbol=symbol,
                    volume=amount,
                    options={'comment': 'sell', 'clientId': client_id, 'takeProfit': tp_price, 'stopLoss': sl_price}
                )
            print(f"Trade executed: {result}")
        except Exception as err:
            logging.error(f"Error placing trade: {err}")

    def is_market_open(self, symbol, server_time):
        """Check if the market is open based on server time and symbol."""
        weekday = server_time.weekday()
        hour = server_time.hour

        if weekday in (5, 6):  # Saturday (5) and Sunday (6)
            return False
        # Market hours, adjust as needed for Forex market
        elif 0 < hour < 22:  
            return True
        else:
            return False

    async def trade_based_on_predictions(self):
        """Make trading decisions based on LSTM model predictions."""
        for symbol in self.symbols:
            # Get predictions from the LSTM model
            predictions = make_predictions(symbol)
            if predictions is not None:
                latest_prediction = predictions[-1]  # Get the latest prediction
                # Define trade logic based on prediction
                trade_type = "ORDER_TYPE_BUY" if latest_prediction > 0 else "ORDER_TYPE_SELL"
                amount = 0.1  # Define your amount

                # Execute the trade
                await self.execute_trade(symbol, trade_type, amount)

    async def run(self):
        """Main function to connect and execute trades based on predictions."""
        await self.connect_account()
        await self.fetch_metrics()
        await self.fetch_recent_trades()
        await self.trade_based_on_predictions()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Example usage
if __name__ == "__main__":
    import yaml
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    trader = MetaApiTrader(config)
    asyncio.run(trader.run())
