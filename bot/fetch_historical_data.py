import os
import yaml
import asyncio
from datetime import datetime, timedelta
import csv
import pandas as pd
import yfinance as yf
from metaapi_cloud_sdk import MetaApi

# Load configuration from config.yaml
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define your authentication tokens and account IDs from config.yaml
mt4_token = config['mt4_token']
mt4_account_id = config['mt4_account_id']
symbols = config['assets']
domain = os.getenv('DOMAIN') or 'agiliumtrade.agiliumtrade.ai'
resample_frequency = config.get('resample_frequency', '5min')  # Default to 5 minutes if not specified

# Define the period and interval for yfinance data
yf_period = '1y'  # Last 1 year
yf_interval = '1h'  # 1-hour interval

# Mapping of MetaTrader symbols to Yahoo Finance symbols
yf_symbols = {
    'XAUUSD': 'GC=F',  # Gold futures
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X'
}

def fetch_yfinance_data(symbol):
    print(f"Fetching data for {symbol} from Yahoo Finance...")
    yf_symbol = yf_symbols.get(symbol, symbol)
    data = yf.download(yf_symbol, period=yf_period, interval=yf_interval)
    return data

async def retrieve_historical_candles(api, account_id, symbol):
    try:
        account = await api.metatrader_account_api.get_account(account_id)

        # Wait until account is deployed and connected to broker
        if account.state != 'DEPLOYED':
            await account.deploy()
        if account.connection_status != 'CONNECTED':
            await account.wait_connected()

        # Retrieve last 10K 1m candles
        pages = 10
        start_time = None
        candles = []
        for i in range(pages):
            new_candles = await account.get_historical_candles(symbol, '1m', start_time)
            candles.extend(new_candles)
            if new_candles:
                start_time = new_candles[0]['time']
                start_time = start_time.replace(minute=start_time.minute - 1)
        return candles

    except Exception as err:
        print(f"Error retrieving candles for {symbol}: {err}")
        return []

def resample_data(data, frequency):
    df = data.copy()
    df.index = pd.to_datetime(df.index)
    
    resampled = df.resample(frequency).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    return resampled

async def fetch_data_for_symbol(symbol, mt4_api):
    yf_data = fetch_yfinance_data(symbol)
    candles = await retrieve_historical_candles(mt4_api, mt4_account_id, symbol)
    return symbol, yf_data, candles

async def main():
    mt4_api = MetaApi(mt4_token, {'domain': domain})

    combined_data = {}

    # Retrieve data for each symbol concurrently
    tasks = [fetch_data_for_symbol(symbol, mt4_api) for symbol in symbols]
    results = await asyncio.gather(*tasks)

    for symbol, yf_data, candles in results:
        combined_data[symbol] = {
            'yf_data': yf_data,
            'candles': candles
        }

    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save combined data to CSV files
    for symbol, data in combined_data.items():
        file_path = f'data/{symbol}.csv'
        
        # Resample yfinance data
        resampled_yf_data = resample_data(data['yf_data'], resample_frequency)
        
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write headers
            writer.writerow(['Type', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Write resampled yfinance data
            for index, row in resampled_yf_data.iterrows():
                writer.writerow(['YFinance', index, row['Open'], row['High'], row['Low'], row['Close'], row['Volume']])
            
            # Write MetaAPI candle data
            for candle in data['candles']:
                writer.writerow(['Candle', candle['time'], candle['open'], candle['high'], candle['low'], candle['close'], candle.get('volume', '')])

    print("Data has been fetched, resampled, and saved successfully.")

if __name__ == "__main__":
    asyncio.run(main())
