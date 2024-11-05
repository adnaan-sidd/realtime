from metaapi_cloud_sdk import MetaApi
import asyncio
import logging
import json
import numpy as np
from models.lstm_model import make_predictions
import uuid
from datetime import datetime, timedelta

class MetaApiTrader:
    def __init__(self, config):
        self.token = config['api_keys']['mt4_token']
        self.account_id = config['api_keys']['mt4_account_id']
        self.api = MetaApi(self.token)
        self.account = None
        self.symbols = config['assets']
        self.take_profit = config['risk_management'].get('take_profit', 0.02)
        self.stop_loss = config['risk_management'].get('stop_loss', 0.01)
        self.max_risk = config['risk_management'].get('max_position_size', 0.02)  # 2% of balance

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

    async def get_account_info(self):
        """Get account information through RPC connection."""
        try:
            connection = self.account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()
            
            account_info = await connection.get_account_information()
            return account_info
        except Exception as err:
            logging.error(f"Error getting account information: {err}")
            return None

    async def calculate_volume(self, balance):
        """Calculate trade volume based on account balance and risk."""
        risk_amount = balance * self.max_risk
        volume = round(risk_amount / 1000, 2)
        return max(0.01, min(volume, 1.0))

    async def monitor_trades(self):
        """Continuously monitor trades to manage TP, SL, and hold strategies."""
        while True:
            try:
                connection = self.account.get_rpc_connection()
                await connection.connect()
                await connection.wait_synchronized()
                
                positions = await connection.get_positions()
                for position in positions:
                    symbol = position['symbol']
                    price_data = await connection.get_symbol_price(symbol)
                    
                    if symbol == 'XAUUSD':
                        if position['profit'] > 10:
                            await connection.close_position(position['id'])
                    else:
                        market_end_time = datetime.utcnow().replace(hour=21, minute=55, second=0)
                        if datetime.utcnow() >= market_end_time or position['profit'] > 5:
                            await connection.close_position(position['id'])
                            
                await asyncio.sleep(600)
            except Exception as e:
                logging.error(f"Error monitoring trades: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying if there's an error

    async def execute_trade(self, symbol, trade_type, balance):
        """Execute a market order based on predictions and dynamic volume calculation."""
        try:
            volume = await self.calculate_volume(balance)
            connection = self.account.get_rpc_connection()
            await connection.connect()
            await connection.wait_synchronized()

            client_id = f"TE_{symbol}_{uuid.uuid4().hex[:8]}"
            price_data = await connection.get_symbol_price(symbol)
            current_price = price_data['bid'] if trade_type == "ORDER_TYPE_SELL" else price_data['ask']
            
            tp_price = current_price * (1 + self.take_profit) if trade_type == "ORDER_TYPE_BUY" else current_price * (1 - self.take_profit)
            sl_price = current_price * (1 - self.stop_loss) if trade_type == "ORDER_TYPE_BUY" else current_price * (1 + self.stop_loss)
            
            if trade_type == "ORDER_TYPE_BUY":
                await connection.create_market_buy_order(
                    symbol=symbol,
                    volume=volume,
                    options={'comment': 'buy', 'clientId': client_id, 'takeProfit': tp_price, 'stopLoss': sl_price}
                )
            else:
                await connection.create_market_sell_order(
                    symbol=symbol,
                    volume=volume,
                    options={'comment': 'sell', 'clientId': client_id, 'takeProfit': tp_price, 'stopLoss': sl_price}
                )
            
            logging.info(f"Trade executed for {symbol}: {trade_type} at volume {volume}")
        except Exception as err:
            logging.error(f"Error executing trade for {symbol}: {err}")

    async def trade_based_on_predictions(self):
        """Execute trades based on LSTM model predictions and manage account balance."""
        try:
            account_info = await self.get_account_info()
            if not account_info:
                logging.error("Could not get account information")
                return
                
            balance = account_info['balance']

            for symbol in self.symbols:
                try:
                    predictions = make_predictions(symbol)
                    if predictions is None or not isinstance(predictions, (np.ndarray, list)) or len(predictions) == 0:
                        continue
                    
                    last_prediction = predictions[-1]
                    connection = self.account.get_rpc_connection()
                    await connection.connect()
                    await connection.wait_synchronized()
                    
                    price_data = await connection.get_symbol_price(symbol)
                    current_ask = price_data['ask']
                    current_bid = price_data['bid']

                    if last_prediction > current_ask:
                        logging.info(f"Signal for {symbol}: Predicted to increase, executing BUY")
                        await self.execute_trade(symbol, "ORDER_TYPE_BUY", balance)
                    elif last_prediction < current_bid:
                        logging.info(f"Signal for {symbol}: Predicted to decrease, executing SELL")
                        await self.execute_trade(symbol, "ORDER_TYPE_SELL", balance)
                except Exception as e:
                    logging.error(f"Error processing trade for {symbol}: {e}")
        except Exception as e:
            logging.error(f"Error in trade_based_on_predictions: {e}")

    async def run(self):
        """Main function to connect, execute trades based on predictions, and monitor trades."""
        await self.connect_account()
        trading_task = asyncio.create_task(self.trade_based_on_predictions())
        monitoring_task = asyncio.create_task(self.monitor_trades())
        
        try:
            await asyncio.gather(trading_task, monitoring_task)
        except Exception as e:
            logging.error(f"Error in main run loop: {e}")
            
# Configure logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    import yaml
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    trader = MetaApiTrader(config)
    asyncio.run(trader.run())
