import MetaTrader5 as mt5
import logging
import yaml
import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, time as dtime
from typing import Dict, Any
from models.lstm_model import make_predictions
import threading
import pytz  # Import pytz for timezone handling


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class MT5Trader:
    """Class to handle MT5 trading operations."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.logger = self._setup_logging()
        self.credentials = self.get_mt5_credentials()
        self.initialize_mt5()

    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Configure and return logger instance."""
        os.makedirs('logs', exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(f'logs/mt5_trader_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

        return logger

    def load_config(self) -> Dict[str, Any]:
        """Load and validate YAML configuration."""
        if not os.path.exists(self.config_path):
            raise ConfigurationError(f"Config file not found at {self.config_path}")

        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            self._validate_config(config)
            return config
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters."""
        required_fields = ['mt5_credentials', 'assets', 'email', 'trading_preferences']
        mt5_required_fields = ['account_number', 'password', 'server']
        email_required_fields = ['sender', 'recipient', 'smtp_server', 'smtp_port', 'password']

        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ConfigurationError(f"Missing required fields in config: {missing_fields}")

        missing_mt5_fields = [field for field in mt5_required_fields if field not in config['mt5_credentials']]
        if missing_mt5_fields:
            raise ConfigurationError(f"Missing required MT5 credentials: {missing_mt5_fields}")

        missing_email_fields = [field for field in email_required_fields if field not in config['email']]
        if missing_email_fields:
            raise ConfigurationError(f"Missing required email configuration: {missing_email_fields}")

    def get_mt5_credentials(self) -> Dict[str, Any]:
        """Get MT5 credentials from config."""
        try:
            return {
                'login': int(self.config['mt5_credentials']['account_number']),
                'password': self.config['mt5_credentials']['password'],
                'server': self.config['mt5_credentials']['server']
            }
        except (KeyError, TypeError, ValueError) as e:
            raise ConfigurationError(f"Invalid MT5 credentials configuration: {e}")

    def initialize_mt5(self) -> None:
        """Initialize MT5 connection with a retry mechanism."""
        retries = 3
        for attempt in range(retries):
            if mt5.initialize(login=self.credentials['login'], password=self.credentials['password'], server=self.credentials['server']):
                self.logger.info("MT5 connection initialized successfully")
                return
            else:
                error_code = mt5.last_error()
                self.logger.error(f"Failed to initialize MT5: {error_code}, attempt {attempt + 1} of {retries}")
                time.sleep(5)  # Wait before retrying
        raise RuntimeError(f"Failed to initialize MT5 after {retries} attempts")

    def shutdown_mt5(self) -> None:
        """Shutdown MT5 connection."""
        mt5.shutdown()
        self.logger.info("MT5 connection closed")

    def send_email(self, subject: str, body: str) -> None:
        """Send an email notification."""
        config = self.config['email']
        msg = MIMEMultipart()
        msg['From'] = config['sender']
        msg['To'] = config['recipient']
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['sender'], config['password'])
                server.sendmail(config['sender'], config['recipient'], msg.as_string())
            self.logger.info("Email notification sent successfully")
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")

    def get_account_balance(self) -> float:
        """Get the current account balance."""
        account_info = mt5.account_info()
        if account_info is None:
            self.logger.error("Failed to get account info")
            return None
        return account_info.balance

    def calculate_position_size(self, risk: float, stop_loss: float, account_balance: float) -> float:
        """Calculate the position size based on risk management."""
        risk_amount = account_balance * risk
        position_size = risk_amount / stop_loss
        return min(max(position_size, self.config['risk_management']['base_position_size']),
                   self.config['risk_management']['max_position_size'])

    def is_market_open(self, symbol: str) -> bool:
        """Check if the market for the given symbol is open."""
        market_info = mt5.symbol_info(symbol)
        if market_info is None or not market_info.visible:
            self.logger.error(f"Symbol {symbol} not found or not visible.")
            return False

        tick_info = mt5.symbol_info_tick(symbol)
        if tick_info is None:
            self.logger.error(f"Failed to get tick information for {symbol}.")
            return False

        server_time = tick_info.time
        current_time = datetime.fromtimestamp(server_time)

        timezone_str = self.config['trading_preferences'].get('timezone', 'UTC')
        start_hour = self.config['trading_preferences'].get('start_hour', 0)
        end_hour = self.config['trading_preferences'].get('end_hour', 23)

        try:
            timezone = pytz.timezone(timezone_str)
        except pytz.UnknownTimeZoneError:
            self.logger.error(f"Unknown timezone: {timezone_str}. Defaulting to UTC.")
            timezone = pytz.utc

        current_time = current_time.astimezone(timezone)

        return dtime(start_hour, 0) <= current_time.time() <= dtime(end_hour, 0)

    def execute_trade(self, symbol: str, action: str, duration: int) -> bool:
        """Execute a trade on MT5 with risk management."""
        if action not in ['buy', 'sell']:
            self.logger.error(f"Invalid trade action: {action}")
            return False

        if not self.is_market_open(symbol):
            self.logger.error(f"Market closed for symbol {symbol}. Cannot execute trade.")
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol {symbol} not found")
            return False

        if not symbol_info.visible:
            self.logger.info(f"Symbol {symbol} is not visible, trying to add it.")
            if not mt5.symbol_select(symbol):
                self.logger.error(f"Failed to add symbol {symbol} to the market watch.")
                return False

        price = mt5.symbol_info_tick(symbol).ask if action == 'buy' else mt5.symbol_info_tick(symbol).bid
        stop_loss_percentage = self.config['risk_management']['stop_loss']
        take_profit_percentage = self.config['risk_management']['take_profit']

        stop_loss = price * (1 - stop_loss_percentage) if action == 'buy' else price * (1 + stop_loss_percentage)
        take_profit = price * (1 + take_profit_percentage) if action == 'buy' else price * (1 - take_profit_percentage)

        account_balance = self.get_account_balance()
        if account_balance is None:
            return False
        position_size = self.calculate_position_size(
            self.config['risk_management']['max_risk_per_trade'],
            stop_loss_percentage,
            account_balance
        )

        order_type = mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': position_size,
            'type': order_type,
            'price': price,
            'sl': stop_loss,
            'tp': take_profit,
            'deviation': 10,
            'magic': 234000,
            'comment': 'Python script open',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Trade execution failed: {result.retcode} - {result.comment}")
            self.send_email("Trade Execution Failed", f"Trade execution failed: {result.retcode} - {result.comment}")
            return False

        self.logger.info(f"Trade executed successfully: {action} {position_size} {symbol} at {price} SL: {stop_loss}, TP: {take_profit}")

        threading.Thread(target=self._wait_and_close_trade, args=(symbol, order_type, position_size, duration)).start()

        return True

    def _wait_and_close_trade(self, symbol: str, order_type: int, position_size: float, duration: int) -> None:
        """Wait for the specified duration and then close the trade."""
        time.sleep(duration)
        self.close_trade(symbol, order_type, position_size)

    def close_trade(self, symbol: str, order_type: int, position_size: float) -> None:
        """Close a trade on MT5."""
        close_order_type = mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        close_request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': position_size,
            'type': close_order_type,
            'deviation': 10,
            'magic': 234000,
            'comment': 'Python script close',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(close_request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Trade closure failed: {result.retcode} - {result.comment}")
            self.send_email("Trade Closure Failed", f"Trade closure failed: {result.retcode} - {result.comment}")
        else:
            self.logger.info(f"Trade closed successfully for {symbol}")

    def trade_on_predictions(self) -> None:
        """Execute trades based on model predictions."""
        for symbol in self.config['assets']:
            prediction, duration = make_predictions(symbol)
            if prediction is not None:
                if prediction[0] > 0:
                    action = 'buy'
                else:
                    action = 'sell'
                self.execute_trade(symbol, action, duration)
            else:
                self.logger.warning(f"No prediction available for {symbol}")

    def reload_config(self) -> None:
        """Reload the configuration from the YAML file."""
        self.config = self.load_config()
        self.credentials = self.get_mt5_credentials()
        self.logger.info("Configuration reloaded successfully")


if __name__ == "__main__":
    trader = MT5Trader('config/config.yaml')
    try:
        trader.trade_on_predictions()
    finally:
        trader.shutdown_mt5()
