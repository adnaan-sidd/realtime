import MetaTrader5 as mt5
import numpy as np
import tensorflow as tf
import logging
import yaml
import os
import pandas as pd

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except yaml.YAMLError as exc:
    if hasattr(exc, 'problem_mark'):
        mark = exc.problem_mark
        logger.error(f"Error in YAML file at line {mark.line + 1}, column {mark.column + 1}")
    else:
        logger.error("Error in YAML file.")
    logger.error(exc)
    quit()

mt5_credentials = config['mt5_credentials']
risk_management = config['risk_management']

# Connect to MetaTrader 5
if not mt5.initialize():
    logger.error("Failed to initialize MetaTrader 5, error code: %s", mt5.last_error())
    quit()

# Login to your account
account = int(mt5_credentials['account_number'])
password = mt5_credentials['password']
server = mt5_credentials['server']

if not mt5.login(account, password, server):
    logger.error("Failed to login to MetaTrader 5, error code: %s", mt5.last_error())
    mt5.shutdown()
    quit()

def get_account_info():
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info, error code: %s", mt5.last_error())
        return None
    return account_info

def calculate_position_size(balance, risk_management):
    base_position_size = risk_management['base_position_size']
    max_position_size = risk_management['max_position_size']
    position_size = min(base_position_size, balance * 0.01)
    return min(position_size, max_position_size)

def manage_open_positions(symbol, prediction, risk_management):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for position in positions:
            if prediction > position.price_open:
                tp_price = position.price_open * (1 + risk_management['take_profit'])
                sl_price = position.price_open * (1 - risk_management['stop_loss'])
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position.ticket,
                    "sl": sl_price,
                    "tp": tp_price,
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    logger.error(f"Failed to modify position {position.ticket}, retcode={result.retcode}")
                else:
                    logger.info(f"Modified position {position.ticket}, TP={tp_price}, SL={sl_price}")

def execute_trade(symbol, prediction, risk_management):
    account_info = get_account_info()
    if account_info is None:
        return

    balance = account_info.balance
    position_size = calculate_position_size(balance, risk_management)

    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        logger.error(f"Failed to get tick data for {symbol}, error code: %s", mt5.last_error())
        return

    price = tick.ask if prediction > tick.ask else tick.bid
    trade_type = mt5.ORDER_TYPE_BUY if prediction > tick.ask else mt5.ORDER_TYPE_SELL

    # Try different filling modes until one succeeds
    filling_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
    for filling_mode in filling_modes:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position_size,
            "type": trade_type,
            "price": price,
            "deviation": 10,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        logger.info(f"Sending order for {symbol} with filling mode {filling_mode}: {request}")
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Order sent successfully for {symbol}, {result}")
            return
        elif result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            logger.warning(f"Order failed for {symbol} with filling mode {filling_mode}, retcode={result.retcode}, result={result}")
        else:
            logger.error(f"Order send failed for {symbol} with filling mode {filling_mode}, retcode={result.retcode}, result={result}")

    logger.error(f"Failed to send order for {symbol} using any of the supported filling modes.")

# List of symbols to trade
symbols = config['assets']

try:
    for symbol in symbols:
        model_path = f'models/{symbol}_best_model.keras'
        if not os.path.exists(model_path):
            logger.warning(f"No model found for {symbol}. Please train the model first.")
            continue

        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            continue

        # Load preprocessed data
        data_dir = config['system']['data_directory']
        x_path = os.path.join(data_dir, f'{symbol}_X.npy')
        y_path = os.path.join(data_dir, f'{symbol}_y.npy')

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            logger.error(f"Preprocessed data files not found for {symbol}: {x_path}, {y_path}")
            continue

        X = np.load(x_path)
        y = np.load(y_path)

        # Select the last 60 rows for prediction
        X_latest = X[-1].reshape(1, 60, X.shape[2])

        # Make prediction
        prediction = model.predict(X_latest)
        logger.info(f"Prediction for {symbol}: {prediction[0][0]}")

        manage_open_positions(symbol, prediction[0][0], risk_management)
        execute_trade(symbol, prediction[0][0], risk_management)
finally:
    mt5.shutdown()