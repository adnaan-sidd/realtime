from yahoo_fin import stock_info as si
import os
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

assets = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F"]
data_dir = "data/yfinance"
os.makedirs(data_dir, exist_ok=True)
date_today = datetime.now().strftime("%Y-%m-%d")
end_date = datetime.now()
start_date = end_date - timedelta(days=2*365)

def check_missing_data(data, asset):
    if data.isnull().values.any():
        print(f"Warning: Missing data in {asset}")
    else:
        print(f"No missing data found in {asset}")

def fetch_and_save_data(asset):
    asset_name = "gold" if asset == "GC=F" else asset.replace("=X", "")
    file_path = f"{data_dir}/{asset_name}_yf.csv"

    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        last_date = existing_data.index[-1].date()
        if last_date < datetime.now().date():
            data = si.get_data(asset, start_date=last_date.strftime("%m/%d/%Y"), end_date=end_date.strftime("%m/%d/%Y"))
            updated_data = pd.concat([existing_data, data])
            updated_data = updated_data.loc[~updated_data.index.duplicated(keep='last')]
            updated_data.to_csv(file_path)
            print(f"Updated data for {asset} saved to {file_path}")
        else:
            print(f"No update needed for {asset}, already up-to-date.")
    else:
        data = si.get_data(asset, start_date=start_date.strftime("%m/%d/%Y"), end_date=end_date.strftime("%m/%d/%Y"))
        if data.empty:
            print(f"No data fetched for {asset}. Verify if the symbol is correct or supported.")
            return
        check_missing_data(data, asset)
        data.to_csv(file_path)
        print(f"Data for {asset} saved to {file_path}")

def fetch_yfinance_data():
    with ThreadPoolExecutor() as executor:
        executor.map(fetch_and_save_data, assets)

fetch_yfinance_data()

