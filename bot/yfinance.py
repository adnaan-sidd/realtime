import logging
from yahoo_fin import stock_info as si
import os
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AssetInfo:
    """Store asset information and mapping."""
    symbol: str
    name: str
    
    @classmethod
    def create(cls, symbol: str) -> 'AssetInfo':
        """Create AssetInfo instance with proper name mapping."""
        name = "gold" if symbol == "GC=F" else symbol.replace("=X", "")
        return cls(symbol=symbol, name=name)

class DataManager:
    """Handles data storage and updates for assets."""
    def __init__(self, base_dir: str = "data/yfinance"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
    def get_file_path(self, asset_info: AssetInfo) -> str:
        """Get the file path for an asset."""
        return os.path.join(self.base_dir, f"{asset_info.name}_yf.csv")
    
    def get_last_date(self, asset_info: AssetInfo) -> Optional[datetime]:
        """Get the last date in the existing file."""
        file_path = self.get_file_path(asset_info)
        if not os.path.exists(file_path):
            return None
            
        try:
            existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if len(existing_data) == 0:
                return None
            return existing_data.index[-1]
        except Exception as e:
            logger.error(f"Error reading file for {asset_info.symbol}: {e}")
            return None
    
    def update_data(self, asset_info: AssetInfo, new_data: pd.DataFrame) -> bool:
        """Update existing file with new data."""
        try:
            file_path = self.get_file_path(asset_info)
            if os.path.exists(file_path):
                existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                updated_data = pd.concat([existing_data, new_data])
                updated_data = updated_data.loc[~updated_data.index.duplicated(keep='last')]
            else:
                updated_data = new_data
                
            updated_data.to_csv(file_path)
            logger.info(f"Updated {asset_info.symbol} with {len(new_data)} new records")
            return True
            
        except Exception as e:
            logger.error(f"Error updating data for {asset_info.symbol}: {e}")
            return False

def check_missing_data(data: pd.DataFrame, asset_info: AssetInfo) -> bool:
    """Check for missing data in the DataFrame."""
    if data.isnull().values.any():
        logger.warning(f"Missing data detected in {asset_info.symbol}")
        return True
    return False

def fetch_asset_data(
    asset_info: AssetInfo,
    data_manager: DataManager,
    lookback_days: int = 730
) -> bool:
    """Fetch and update data for a single asset."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        logger.info(f"Fetching data for {asset_info.symbol} from {start_date} to {end_date}")
        
        data = si.get_data(
            asset_info.symbol,
            start_date=start_date.strftime("%m/%d/%Y"),
            end_date=end_date.strftime("%m/%d/%Y")
        )
        
        if data.empty:
            logger.warning(f"No data fetched for {asset_info.symbol}")
            return True  # Not necessarily an error
            
        check_missing_data(data, asset_info)
        return data_manager.update_data(asset_info, data)
        
    except Exception as e:
        logger.error(f"Error fetching data for {asset_info.symbol}: {e}")
        return False

def update_all_assets(
    assets: List[str],
    max_workers: int = 5
) -> None:
    """Update data for all assets once."""
    data_manager = DataManager()
    asset_infos = [AssetInfo.create(symbol) for symbol in assets]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_asset_data, asset_info, data_manager
            ): asset_info
            for asset_info in asset_infos
        }
        
        for future in as_completed(futures):
            asset_info = futures[future]
            try:
                success = future.result()
                if not success:
                    logger.warning(f"Failed to update {asset_info.symbol}")
            except Exception as e:
                logger.error(f"Task failed for {asset_info.symbol}: {e}")

def continuous_data_update(assets: List[str], lookback_days: int = 730) -> None:
    """Continuously update historical price data for the specified assets."""
    logger.info("Starting continuous data update...")
    update_all_assets(assets, max_workers=5)

if __name__ == "__main__":
    assets = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F"]
    
    # Update all assets once
    update_all_assets(
        assets=assets
    )