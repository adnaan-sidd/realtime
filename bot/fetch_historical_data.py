import os
import yaml
import asyncio
from datetime import datetime
import csv
from metaapi_cloud_sdk import MetaApi

# Load configuration from config.yaml
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define your authentication tokens and account IDs from config.yaml
mt5_token = config['mt5_token']
mt4_token = config['mt4_token']
mt5_account_id = config['mt5_account_id']
mt4_account_id = config['mt4_account_id']
symbols = config['assets']
domain = os.getenv('DOMAIN') or 'agiliumtrade.agiliumtrade.ai'

async def retrieve_historical_ticks(api, account_id, symbol):
    try:
        account = await api.metatrader_account_api.get_account(account_id)

        # Wait until account is deployed and connected to broker
        if account.state != 'DEPLOYED':
            await account.deploy()
        if account.connection_status != 'CONNECTED':
            await account.wait_connected()

        # Retrieve last 10K ticks
        pages = 10
        start_time = datetime.fromtimestamp(datetime.now().timestamp() - 7 * 24 * 60 * 60)
        offset = 0
        ticks = []
        for i in range(pages):
            new_ticks = await account.get_historical_ticks(symbol, start_time, offset)
            ticks.extend(new_ticks)
            if new_ticks:
                start_time = new_ticks[-1]['time']
                offset = 0
                while offset < len(new_ticks) and new_ticks[-1 - offset]['time'].timestamp() == start_time.timestamp():
                    offset += 1
        return ticks

    except Exception as err:
        return []

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
                start_time.replace(minute=start_time.minute - 1)
        return candles

    except Exception as err:
        return []

async def fetch_data_for_symbol(symbol, mt5_api, mt4_api):
    ticks = await retrieve_historical_ticks(mt5_api, mt5_account_id, symbol)
    candles = await retrieve_historical_candles(mt4_api, mt4_account_id, symbol)
    return symbol, ticks, candles

async def main():
    mt5_api = MetaApi(mt5_token, {'domain': domain})
    mt4_api = MetaApi(mt4_token, {'domain': domain})

    combined_data = {}

    # Retrieve data for each symbol concurrently
    tasks = [fetch_data_for_symbol(symbol, mt5_api, mt4_api) for symbol in symbols]
    results = await asyncio.gather(*tasks)

    for symbol, ticks, candles in results:
        combined_data[symbol] = {
            'ticks': ticks,
            'candles': candles
        }

    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save combined data to CSV files
    for symbol, data in combined_data.items():
        file_path = f'data/{symbol}.csv'
        
        # Check if file exists, if not create it and write headers
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write headers if file does not exist
            if not file_exists:
                writer.writerow(['Type', 'Time', 'Bid', 'Ask', 'Volume'])
            
            # Write tick data
            for tick in data['ticks']:
                writer.writerow(['Tick', tick['time'], tick.get('bid', ''), tick.get('ask', ''), tick.get('volume', '')])
            
            # Write candle data
            for candle in data['candles']:
                writer.writerow(['Candle', candle['time'], candle['open'], candle['high'], candle['low'], candle['close'], candle.get('volume', '')])

    print("Data has been fetched, combined, and saved successfully.")

asyncio.run(main())

