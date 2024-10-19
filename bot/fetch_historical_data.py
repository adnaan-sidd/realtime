import os
import pandas as pd
import yfinance as yf
import requests
import asyncio
from metaapi_cloud_sdk import MetaApi
import logging
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)

class DataFetcher:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.metaapi_token = config['metaapi_token']
        self.metaapi_account_id = config['metaapi_account_id']
        self.twelve_data_key = config['twelve_data_key']  # Use Twelve Data API key here
        self.assets = config['assets']
        self.data_folder = 'data'
        os.makedirs(self.data_folder, exist_ok=True)

    # Fetch historical data using MetaAPI for candles
    async def fetch_metaapi_candles(self, symbol, timeframe, start_time, end_time):
        try:
            logging.info(f"Connecting to MetaAPI with token: {self.metaapi_token}")
            api = MetaApi(self.metaapi_token)
            account = await api.metatrader_account_api.get_account(self.metaapi_account_id)

            if not account:
                raise Exception(f"MetaAPI Account not found for symbol {symbol}")

            logging.info(f"Fetching historical candle data for {symbol} from MetaAPI...")
            history = await account.get_historical_candles(symbol, timeframe, start_time, end_time)

            if history:
                self.save_to_csv(history, symbol, 'MetaAPI_Candles')
                return history
            else:
                logging.warning(f"No historical candle data returned for {symbol} from MetaAPI.")
                return None
        except Exception as e:
            logging.error(f"MetaAPI candle fetch failed for {symbol}: {str(e)}")
            return None

    # Fetch historical ticks using MetaAPI
    async def fetch_metaapi_ticks(self, symbol, start_time):
        try:
            api = MetaApi(self.metaapi_token)
            account = await api.metatrader_account_api.get_account(self.metaapi_account_id)

            if not account:
                raise Exception(f"MetaAPI Account not found for symbol {symbol}")

            logging.info(f"Fetching historical tick data for {symbol} from MetaAPI...")
            ticks = await account.get_historical_ticks(symbol, start_time, 0)

            if ticks:
                self.save_to_csv(ticks, symbol, 'MetaAPI_Ticks')
                return ticks
            else:
                logging.warning(f"No historical tick data returned for {symbol} from MetaAPI.")
                return None
        except Exception as e:
            logging.error(f"MetaAPI tick fetch failed for {symbol}: {str(e)}")
            return None

    # Fetch historical data using Twelve Data API
    def fetch_twelve_data(self, symbol):
        try:
            logging.info(f"Connecting to Twelve Data with key: {self.twelve_data_key}")
            url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&apikey={self.twelve_data_key}&outputsize=500"
            response = requests.get(url)
            data = response.json()

            if 'values' in data:
                df = pd.DataFrame(data['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                df.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
                self.save_to_csv(df, symbol, 'TwelveData')
                return df
            else:
                logging.warning(f"Twelve Data returned no data for {symbol}: {data.get('message', 'No message')}")
                return None
        except Exception as e:
            logging.error(f"Twelve Data fetch failed for {symbol}: {str(e)}")
            return None

    # Fallback to Yahoo Finance if both MetaAPI and Twelve Data fail
    def fetch_yahoo_data(self, symbol, start_date, end_date):
        try:
            symbol_yahoo = f"{symbol}=X"
            logging.info(f"Fetching historical data for {symbol} from Yahoo Finance...")
            data = yf.download(symbol_yahoo, start=start_date, end=end_date, interval='1h')
            if not data.empty:
                self.save_to_csv(data, symbol, 'YahooFinance')
                return data
            else:
                logging.warning(f"Yahoo Finance returned no data for {symbol}.")
                return None
        except Exception as e:
            logging.error(f"Yahoo Finance fetch failed for {symbol}: {str(e)}")
            return None

    def save_to_csv(self, data, symbol, source):
        output_path = os.path.join(self.data_folder, f"{symbol}_data_{source}.csv")
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path)
        else:
            df = pd.DataFrame(data)
            df.to_csv(output_path)
        logging.info(f"Data for {symbol} saved from {source} to {output_path}")

    async def fetch_all_assets(self, start_time, end_time, start_date, end_date):
        for asset in self.assets:
            logging.info(f"Fetching candle data for {asset}...")
            # Try fetching from MetaAPI
            candles = await self.fetch_metaapi_candles(asset, '1H', start_time, end_time)
            if not candles:
                # Fallback to Twelve Data
                candles = self.fetch_twelve_data(asset)
            if not candles:
                # Fallback to Yahoo Finance
                self.fetch_yahoo_data(asset, start_date, end_date)

            logging.info(f"Fetching tick data for {asset}...")
            # Try fetching ticks from MetaAPI
            ticks = await self.fetch_metaapi_ticks(asset, start_time)
            if not ticks:
                logging.warning(f"No tick data found for {asset}.")

if __name__ == "__main__":
    fetcher = DataFetcher()
    start_time = '2023-01-01T00:00:00Z'
    end_time = '2023-12-31T23:59:59Z'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    asyncio.run(fetcher.fetch_all_assets(start_time, end_time, start_date, end_date))

