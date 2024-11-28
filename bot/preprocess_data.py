import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import os
import pickle
import ast
import yaml
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Setup logging with both file and console handlers
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class Config:
    def __init__(self, config_path: str = 'config/config.yaml'):
        logger.info(f"Loading configuration from {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.validate_config()
        self.setup_directories()
        self.load_config_attributes()

        logger.info("Configuration loaded successfully")

    def validate_config(self):
        required_fields = ['assets', 'model_params', 'preprocessing', 'system']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

    def setup_directories(self):
        os.makedirs(self.config['system']['data_directory'], exist_ok=True)
        os.makedirs(self.config['system']['log_directory'], exist_ok=True)

    def load_config_attributes(self):
        self.assets = self.config['assets']
        self.sequence_length = self.config['model_params']['sequence_length']
        self.feature_columns = self.config['preprocessing']['feature_columns']

class DataLoader:
    def __init__(self, symbol: str, config: Config):
        self.symbol = symbol
        self.config = config
        logger.info(f"Initializing DataLoader for {symbol}")

    def load_and_merge_data(self, sentiment_file: str) -> pd.DataFrame:
        try:
            data_dir = self.config.config['system']['data_directory']
            yfinance_file = self.get_yfinance_file(data_dir)
            logger.info(f"Loading yfinance data from: {yfinance_file}")
            yfinance_data = self.load_yfinance_data(yfinance_file)

            candles_data = self.load_candles_data()
            market_data = pd.concat([yfinance_data, candles_data]).drop_duplicates().sort_index()

            logger.info(f"Loading sentiment data from: {sentiment_file}")
            sentiment_data = self.load_sentiment_data(sentiment_file)
            merged_data = pd.merge(market_data, sentiment_data, left_index=True, right_index=True, how='outer')

            logger.info(f"Successfully merged data for {self.symbol}. Shape: {merged_data.shape}")
            return merged_data

        except Exception as e:
            logger.error(f"Error while loading and merging data: {e}")
            raise

    def get_yfinance_file(self, data_dir):
        if self.symbol == "XAUUSD":
            return os.path.join(data_dir, 'yfinance', 'gold_yf.csv')
        return os.path.join(data_dir, 'yfinance', f'{self.symbol}_yf.csv')

    def load_yfinance_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YFinance data file not found: {file_path}")

        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notnull()]
        df.index = df.index.tz_localize(None)  # Convert to tz-naive
        df.rename(columns=str.capitalize, inplace=True)
        return df

    def load_candles_data(self) -> pd.DataFrame:
        candles_data = []
        candle_timeframes = ['D1', 'H1', 'H4', 'M15']
        data_dir = self.config.config['system']['data_directory']

        for timeframe in candle_timeframes:
            candle_file = os.path.join(data_dir, 'candles', f'{self.symbol}_{timeframe}.csv')
            if os.path.exists(candle_file):
                logger.info(f"Loading candles data from: {candle_file}")
                df = self.load_yfinance_data(candle_file)
                candles_data.append(df)

        return pd.concat(candles_data) if candles_data else pd.DataFrame()

    def load_sentiment_data(self, sentiment_file: str) -> pd.DataFrame:
        if not os.path.exists(sentiment_file):
            raise FileNotFoundError(f"Sentiment data file not found: {sentiment_file}")

        sentiment_data = pd.read_csv(sentiment_file)
        sentiment_data['Date'] = pd.to_datetime(sentiment_data['timestamp'], errors='coerce')
        sentiment_data = sentiment_data[sentiment_data['Date'].notnull()]
        sentiment_data.set_index('Date', inplace=True)
        sentiment_data.index = sentiment_data.index.tz_localize(None)  # Convert to tz-naive
        sentiment_data = self.process_sentiment_scores(sentiment_data)
        return sentiment_data

    def process_sentiment_scores(self, sentiment_data):
        sentiment_data['sentiment_scores'] = sentiment_data['sentiment_scores'].apply(ast.literal_eval)
        sentiment_data['positive'] = sentiment_data['sentiment_scores'].apply(lambda x: x['score'] if x['label'] == 'POSITIVE' else 0)
        sentiment_data['negative'] = sentiment_data['sentiment_scores'].apply(lambda x: x['score'] if x['label'] == 'NEGATIVE' else 0)
        sentiment_data['neutral'] = sentiment_data['sentiment_scores'].apply(lambda x: x['score'] if x['label'] == 'NEUTRAL' else 0)
        sentiment_data.drop(columns=['sentiment_scores'], inplace=True)
        return sentiment_data

class FeatureEngine:
    def __init__(self, config: Config):
        self.config = config

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = self.validate_dataframe(df)

            logger.info("Calculating technical indicators")
            df['Returns'] = df['Close'].pct_change(fill_method=None)

            self.calculate_moving_averages(df)
            self.calculate_rsi(df)
            self.calculate_bollinger_bands(df)
            self.calculate_macd(df)
            self.calculate_atr(df)

            df['Volatility'] = df['Close'].rolling(window=20).std()
            df['Momentum'] = df['Close'] - df['Close'].shift(4)
            df['ROC'] = (df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12) * 100

            logger.info("Technical indicators calculated successfully")
            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'Close' not in df.columns:
            raise ValueError("DataFrame does not contain 'Close' column")
        return df.copy()

    def calculate_moving_averages(self, df: pd.DataFrame):
        short_window = self.config.config['strategies']['ma_crossover']['short_window']
        long_window = self.config.config['strategies']['ma_crossover']['long_window']
        df[f'SMA_{short_window}'] = df['Close'].rolling(window=short_window).mean()
        df[f'SMA_{long_window}'] = df['Close'].rolling(window=long_window).mean()

    def calculate_rsi(self, df: pd.DataFrame):
        rsi_period = self.config.config['strategies']['rsi']['period']
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, df: pd.DataFrame):
        bb_window = self.config.config['strategies']['bollinger']['window']
        bb_std = self.config.config['strategies']['bollinger']['num_std']
        df['BB_middle'] = df['Close'].rolling(window=bb_window).mean()
        df['BB_upper'] = df['BB_middle'] + bb_std * df['Close'].rolling(window=bb_window).std()
        df['BB_lower'] = df['BB_middle'] - bb_std * df['Close'].rolling(window=bb_window).std()

    def calculate_macd(self, df: pd.DataFrame):
        macd_fast = self.config.config['strategies']['macd']['fast']
        macd_slow = self.config.config['strategies']['macd']['slow']
        macd_signal = self.config.config['strategies']['macd']['signal']
        ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
        df['MACD_line'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD_line'].ewm(span=macd_signal, adjust=False).mean()
        df['MACD_histogram'] = df['MACD_line'] - df['MACD_signal']

    def calculate_atr(self, df: pd.DataFrame):
        atr_period = self.config.config['strategies']['atr']['period']
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=atr_period).mean()

class SequencePreparation:
    def __init__(self, config: Config):
        self.config = config
        self.sequence_length = config.sequence_length

    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, dict]:
        try:
            logger.info("Preparing sequences for LSTM training")
            feature_columns = self.config.feature_columns
            
            self.check_missing_columns(data, feature_columns)
            scalers, scaled_data = self.scale_data(data, feature_columns)

            X, y = self.create_sequences(scaled_data, feature_columns)
            logger.info(f"Sequences prepared. X shape: {X.shape}, y shape: {y.shape}")

            return X, y, scalers

        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            raise

    def check_missing_columns(self, data: pd.DataFrame, feature_columns: list):
        missing_columns = [col for col in feature_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")

    def scale_data(self, data: pd.DataFrame, feature_columns: list) -> Tuple[dict, pd.DataFrame]:
        scalers = {}
        scaled_data = pd.DataFrame(index=data.index)

        for column in feature_columns:
            scaler = MinMaxScaler()
            scaled_data[column] = scaler.fit_transform(data[[column]])
            scalers[column] = scaler

        return scalers, scaled_data

    def create_sequences(self, scaled_data: pd.DataFrame, feature_columns: list) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        data_values = scaled_data[feature_columns].values
        close_values = scaled_data['Close'].values

        for i in range(len(data_values) - self.sequence_length):
            X.append(data_values[i:(i + self.sequence_length)])
            y.append(close_values[i + self.sequence_length])

        return np.array(X), np.array(y)

class PreprocessingPipeline:
    def __init__(self, config: Config):
        self.config = config

    def process_symbol(self, symbol: str, sentiment_file: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        try:
            logger.info(f"Starting preprocessing for {symbol}")

            data_loader = DataLoader(symbol, self.config)
            feature_engine = FeatureEngine(self.config)
            sequence_prep = SequencePreparation(self.config)

            # Load data
            merged_data = data_loader.load_and_merge_data(sentiment_file)
            # Calculate technical indicators
            data_with_indicators = feature_engine.calculate_technical_indicators(merged_data)
            # Fill missing values
            data_filled = data_with_indicators.ffill().bfill()

            # Prepare sequences for LSTM
            X, y, scalers = sequence_prep.prepare_sequences(data_filled)

            # Save processed data
            self.save_processed_data(symbol, X, y, scalers)

            logger.info(f"Preprocessing completed for {symbol}. Shape: X: {X.shape}, y: {y.shape}")
            return X, y, scalers

        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
            raise

    def save_processed_data(self, symbol: str, X: np.ndarray, y: np.ndarray, scalers: dict):
        np.save(f'data/{symbol}_X.npy', X)
        np.save(f'data/{symbol}_y.npy', y)

        data_dir = self.config.config['system']['data_directory']
        scaler_path = os.path.join(data_dir, f'{symbol}_scalers.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        logger.info(f"Saved scalers to: {scaler_path}")

    def process_all_symbols(self, sentiment_file: str):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_symbol, symbol, sentiment_file) for symbol in self.config.assets]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing: {e}")

def main():
    try:
        logger.info("Starting preprocessing pipeline...")
        config = Config('config/config.yaml')
        sentiment_file = os.path.join(config.config['system']['data_directory'], 'sentiment_data.csv')
        
        if not os.path.exists(sentiment_file):
            logger.error(f"Sentiment data file not found: {sentiment_file}")
            raise FileNotFoundError(sentiment_file)

        # Cleanup existing data files
        for symbol in config.assets:
            existing_X_file = f'data/{symbol}_X_existing.npy'
            existing_y_file = f'data/{symbol}_y_existing.npy'
            if os.path.exists(existing_X_file):
                logger.info(f"Removing existing file: {existing_X_file}")
                os.remove(existing_X_file)
            if os.path.exists(existing_y_file):
                logger.info(f"Removing existing file: {existing_y_file}")
                os.remove(existing_y_file)

        pipeline = PreprocessingPipeline(config)
        pipeline.process_all_symbols(sentiment_file)
        logger.info("Preprocessing pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()