from metaapi_cloud_sdk import MetaApi
import asyncio
import logging
import json
import numpy as np  # Import numpy for array handling
from models.lstm_model import make_predictions
import uuid

class MetaApiTrader:
    def __init__(self, config):
        self.token = config['api_keys']['mt4_token']
        self.account_id = config['api_keys']['mt4_account_id']
        self.api = MetaApi(self.token)
        self.account = None
        self.symbols = config['assets']
        self.take_profit = config['risk_management'].get('take_profit', 0.02)
        self.stop_loss = config['risk_management'].get('stop_loss', 0.01)

    async def connect_account(self):
        """Connect to the MetaApi account."""
        try:
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)
            if self.account.state != 'DEPLOYED':
                await self.account.deploy()
                logging.info("Deploying account...")

            await self.account.wait_connected()
            logging.info("Account connected to the broker.")
        except Exception as err:
            logging.error(f"Error connecting to account: {err}")

    async def execute_trade(self, symbol, trade_type, volume, target_price):
        """Execute a market order based on predictions."""
        try:
            connection = self.account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()

            client_id = f"TE_{symbol}_{uuid.uuid4().hex[:8]}"

            # Get the current price data
            price_data = await connection.get_symbol_price(symbol)
            current_price = price_data['bid'] if trade_type == "ORDER_TYPE_SELL" else price_data['ask']
            
            if trade_type == "ORDER_TYPE_BUY":
                tp_price = current_price * (1 + self.take_profit)
                sl_price = current_price * (1 - self.stop_loss)
                result = await connection.create_market_buy_order(
                    symbol=symbol,
                    volume=volume,
                    options={'comment': 'buy', 'clientId': client_id, 'takeProfit': tp_price, 'stopLoss': sl_price}
                )
            else:
                tp_price = current_price * (1 - self.take_profit)
                sl_price = current_price * (1 + self.stop_loss)
                result = await connection.create_market_sell_order(
                    symbol=symbol,
                    volume=volume,
                    options={'comment': 'sell', 'clientId': client_id, 'takeProfit': tp_price, 'stopLoss': sl_price}
                )

            logging.info(f"Trade executed for {symbol}: {result}")
            self.update_portfolio_json(symbol, volume, trade_type, current_price, target_price)
        except Exception as err:
            logging.error(f"Error placing trade for {symbol}: {err}")

    async def trade_based_on_predictions(self):
        """Execute trades based on LSTM model predictions."""
        for symbol in self.symbols:
            try:
                predictions = make_predictions(symbol)
                
                if isinstance(predictions, (np.ndarray, list)) and len(predictions) > 0:
                    last_prediction = predictions[-1]
                elif isinstance(predictions, (float, int)):
                    last_prediction = predictions
                else:
                    logging.warning(f"No valid predictions for {symbol}, skipping trade.")
                    continue

                connection = self.account.get_rpc_connection()
                await connection.connect()
                await connection.wait_synchronized()

                price_data = await connection.get_symbol_price(symbol)
                current_bid = price_data['bid']
                current_ask = price_data['ask']

                # Buy if prediction is higher than the ask price, else sell
                if last_prediction > current_ask:
                    logging.info(f"Signal for {symbol}: Predicted to increase, executing BUY at {current_ask}")
                    await self.execute_trade(symbol, "ORDER_TYPE_BUY", 0.1, last_prediction)
                elif last_prediction < current_bid:
                    logging.info(f"Signal for {symbol}: Predicted to decrease, executing SELL at {current_bid}")
                    await self.execute_trade(symbol, "ORDER_TYPE_SELL", 0.1, last_prediction)
            except Exception as e:
                logging.error(f"Error processing trade for {symbol}: {e}")

    def update_portfolio_json(self, symbol, volume, trade_type, entry_price, target_price):
        """Update portfolio status in JSON file for web interface."""
        portfolio_data = {
            "balance": 100000,  # Replace with actual balance fetching code
            "open_trades": [
                {
                    "symbol": symbol,
                    "volume": volume,
                    "trade_type": trade_type,
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "status": "open"
                }
            ],
            "metrics": {
                "take_profit": self.take_profit,
                "stop_loss": self.stop_loss
            }
        }
        with open("data/portfolio.json", "w") as f:
            json.dump(portfolio_data, f, indent=4)

    async def run(self):
        """Main function to connect and execute trades based on predictions."""
        await self.connect_account()
        await self.trade_based_on_predictions()

# Configure logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    import yaml
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    trader = MetaApiTrader(config)
    asyncio.run(trader.run())
