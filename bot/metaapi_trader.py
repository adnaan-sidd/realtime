from metaapi_cloud_sdk import MetaApi, MetaStats
import asyncio
import logging
import numpy as np
import json
from models.lstm_model import make_predictions
import uuid
from datetime import datetime

class MetaApiTrader:
    def __init__(self, config):
        self.token = config['api_keys']['mt4_token']
        self.account_id = config['api_keys']['mt4_account_id']
        self.api = MetaApi(self.token)
        self.meta_stats = MetaStats(self.token)
        self.account = None
        self.connection = None
        self.symbols = config.get('assets', ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'])

        # Risk management parameters from config
        self.take_profit = config['risk_management'].get('take_profit', 0.02)
        self.stop_loss = config['risk_management'].get('stop_loss', 0.01)
        
        # Position sizing parameters
        self.base_position_size = config['risk_management'].get('base_position_size', 0.1)
        self.max_position_size = config['risk_management'].get('max_position_size', 1.0)

    async def connect_account(self):
        """Connect to the MetaApi account."""
        try:
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)

            # Deploy account if not already deployed
            if self.account.state != 'DEPLOYED':
                await self.account.deploy()
                logging.info("Account is being deployed...")
            else:
                logging.info("Account is already deployed.")

            # Wait for the account to connect
            await self.account.wait_connected()
            
            # Establish and store RPC connection
            self.connection = self.account.get_rpc_connection()
            await self.connection.connect()
            await self.connection.wait_synchronized()
            
            logging.info("Account is connected to the broker.")
        except Exception as err:
            logging.error(f"Error connecting to account: {err}")
            raise

    async def check_market_status(self, symbol):
        """Check if the market is open for trading a specific symbol."""
        try:
            if not self.connection:
                raise Exception("No connection established")

            # Get symbol specification
            symbol_spec = await self.connection.get_symbol_specification(symbol)
            
            # Check if trading is allowed for this symbol
            if not symbol_spec.get('trade_allowed', False):
                logging.warning(f"Trading not allowed for {symbol}")
                return False

            # Check if market is open based on trading sessions
            # Most MetaTrader brokers will handle this automatically,
            # so we'll trust the trade_allowed flag
            return symbol_spec.get('trade_allowed', False)

        except Exception as e:
            logging.error(f"Error checking market status for {symbol}: {e}")
            return False

    async def execute_trade(self, symbol, trade_type, amount):
        """Execute a market order with risk management."""
        try:
            if not self.connection:
                raise Exception("No connection established")

            # Check if market is open
            is_market_open = await self.check_market_status(symbol)
            if not is_market_open:
                logging.warning(f"Market is closed or trading not allowed for {symbol}. Trade not executed.")
                return

            # Generate a unique client ID
            client_id = f"TE_{symbol}_{uuid.uuid4().hex[:8]}"

            # Get current market price
            price_data = await self.connection.get_symbol_price(symbol)
            if not price_data:
                logging.error(f"Could not get price data for {symbol}")
                return

            # Calculate take profit and stop loss prices
            open_price = price_data['ask'] if trade_type == "ORDER_TYPE_BUY" else price_data['bid']
            tp_price = open_price * (1 + self.take_profit) if trade_type == "ORDER_TYPE_BUY" else open_price * (1 - self.take_profit)
            sl_price = open_price * (1 - self.stop_loss) if trade_type == "ORDER_TYPE_BUY" else open_price * (1 + self.stop_loss)

            # Execute the trade
            trade_options = {
                'comment': f'auto_{trade_type.lower()}',
                'clientId': client_id,
                'takeProfit': tp_price,
                'stopLoss': sl_price
            }

            if trade_type == "ORDER_TYPE_BUY":
                result = await self.connection.create_market_buy_order(
                    symbol=symbol,
                    volume=amount,
                    options=trade_options
                )
            else:
                result = await self.connection.create_market_sell_order(
                    symbol=symbol,
                    volume=amount,
                    options=trade_options
                )

            logging.info(f"Trade executed for {symbol}: {result}")
            return result

        except Exception as err:
            logging.error(f"Error executing trade for {symbol}: {err}")
            raise

    def calculate_position_size(self, confidence):
        """Calculate the position size based on the confidence of the trading signal."""
        # Ensure confidence is between 0 and 1
        confidence = max(0, min(1, confidence))
        
        # Calculate position size based on confidence
        position_size = self.base_position_size + (self.max_position_size - self.base_position_size) * confidence
        
        # Ensure position size does not exceed max position size
        position_size = min(position_size, self.max_position_size)
        
        return position_size

    async def trade_based_on_signals(self, trading_signals):
        """Execute trades based on provided trading signals."""
        if not trading_signals:
            logging.warning("No trading signals provided")
            return

        for symbol, signal in trading_signals.items():
            if symbol not in self.symbols:
                logging.warning(f"Signal received for unknown symbol: {symbol}")
                continue

            try:
                # Determine trade type based on signal
                trade_type = "ORDER_TYPE_BUY" if signal['action'] == 'buy' else "ORDER_TYPE_SELL"
                
                # Calculate position size based on confidence
                amount = self.calculate_position_size(signal['confidence'])

                # Execute the trade
                result = await self.execute_trade(symbol, trade_type, amount)
                if result:
                    logging.info(f"Successfully executed {trade_type} for {symbol} with amount {amount}")

            except Exception as e:
                logging.error(f"Error executing trade for {symbol}: {e}")
                continue

    async def run(self, trading_signals=None):
        """Main function to connect and execute trades based on signals."""
        try:
            await self.connect_account()
            
            if trading_signals:
                await self.trade_based_on_signals(trading_signals)
            else:
                logging.warning("No trading signals provided, skipping trade execution")
                
        except Exception as e:
            logging.error(f"Error in trading execution: {e}")
            raise
        finally:
            if self.connection:
                try:
                    await self.connection.close()  # Use the correct method to close the connection
                    logging.info("Disconnected from MetaAPI")
                except Exception as e:
                    logging.error(f"Error disconnecting: {e}")

if __name__ == "__main__":
    import yaml
    
    # Load config and run trader
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    trader = MetaApiTrader(config)
    asyncio.run(trader.run())
