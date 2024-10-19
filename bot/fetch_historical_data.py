# fetch_historical_data.py

import os
import pandas as pd
import yfinance as yf
from alpha_vantage.foreignexchange import ForeignExchange
import asyncio
from metaapi_cloud_sdk import MetaApi

class DataFetcher:
    def __init__(self, config_path="config/config.yaml"):
        import yaml
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.metaapi_token = config['metaapi_token']
        self.metaapi_account_id = config['metaapi_account_id']
        self.alpha_vantage_key = config['alpha_vantage_key']
        self.assets = config['assets']
        self.data_folder = 'data'

    # Step 1: Fetch historical data using MetaAPI
    async def fetch_metaapi_data(self, symbol, timeframe, start_time, end_time):
        """Fetch historical data from MetaAPI."""
        try:
            api = MetaApi(self.metaapi_token)
            account = await api.metatrader_account_api.get_account(self.metaapi_account_id)

            if not account:
                raise Exception(f"MetaAPI Account not found for symbol {symbol}")

            history = await account.get_historical_candles(symbol, timeframe, start_time, end_time)
            if history:
                self.save_to_csv(history, symbol, 'MetaAPI')
                return history
        except Exception as e:
            print(f"MetaAPI fetch failed for {symbol}: {str(e)}")
            return None

    # Step 2: Fallback to Alpha Vantage if MetaAPI fails
    def fetch_alpha_vantage_data(self, symbol):
        """Fetch historical data from Alpha Vantage."""
        try:
            fx = ForeignExchange(key=self.alpha_vantage_key)
            symbol_mapped = self.map_asset_to_alpha_vantage(symbol)
            data, _ = fx.get_currency_exchange_intraday(symbol_mapped, interval='60min', outputsize='full')
            df = pd.DataFrame.from_dict(data, orient='index')
            df.columns = ['open', 'high', 'low', 'close']
            df.index = pd.to_datetime(df.index)
            df['volume'] = 0  # Alpha Vantage doesn't provide volume for forex
            self.save_to_csv(df, symbol, 'AlphaVantage')
            return df
        except Exception as e:
            print(f"Alpha Vantage fetch failed for {symbol}: {str(e)}")
            return None

    # Step 3: Fallback to Yahoo Finance (yfinance) if both MetaAPI and Alpha Vantage fail
    def fetch_yahoo_data(self, symbol, start_date, end_date):
        """Fetch historical data from Yahoo Finance."""
        try:
            symbol_yahoo = f"{symbol}=X"  # Convert to Yahoo format, e.g., 'EURUSD=X'
            data = yf.download(symbol_yahoo, start=start_date, end=end_date, interval='1h')
            if not data.empty:
                self.save_to_csv(data, symbol, 'YahooFinance')
                return data
            else:
                print(f"Yahoo Finance returned no data for {symbol}.")
                return None
        except Exception as e:
            print(f"Yahoo Finance fetch failed for {symbol}: {str(e)}")
            return None

    def map_asset_to_alpha_vantage(self, symbol):
        """Map asset symbols for Alpha Vantage (e.g., EURUSD -> EUR/USD)."""
        mapping = {
            'EURUSD': 'EUR/USD',
            'GBPUSD': 'GBP/USD',
            'JPYUSD': 'JPY/USD',
            'XAUUSD': 'XAU/USD'
        }
        return mapping.get(symbol, symbol)

    def save_to_csv(self, data, symbol, source):
        """Save fetched data to a CSV file."""
        output_path = os.path.join(self.data_folder, f"{symbol}_data_{source}.csv")
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path)
        else:
            df = pd.DataFrame(data)
            df.to_csv(output_path)
        print(f"Data for {symbol} saved from {source} to {output_path}")

    async def fetch_all_assets(self, start_time, end_time, start_date, end_date):
        """Fetch data for all assets using MetaAPI first, then fallback to Alpha Vantage or Yahoo Finance."""
        for asset in self.assets:
            print(f"Fetching data for {asset}...")
            # Try fetching from MetaAPI
            history = await self.fetch_metaapi_data(asset, '1H', start_time, end_time)
            if not history:
                # Fallback to Alpha Vantage
                history = self.fetch_alpha_vantage_data(asset)
            if not history:
                # Fallback to Yahoo Finance
                self.fetch_yahoo_data(asset, start_date, end_date)

# Example usage
if __name__ == "__main__":
    fetcher = DataFetcher()
    # Define time range for historical data fetching
    start_time = '2023-01-01T00:00:00Z'
    end_time = '2023-12-31T23:59:59Z'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    asyncio.run(fetcher.fetch_all_assets(start_time, end_time, start_date, end_date))

