import yaml
import logging
import os
import sys
import asyncio
from datetime import datetime

# Add directories to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'news')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'strategies')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'indicators')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_fetchers')))

from news_scraper import NewsScraper
from candles import fetch_candles_data
from yfinance import fetch_yfinance_data
from preprocess_data import preprocess_data
from lstm_model import train_model
from metaapi_trader import MetaApiTrader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_and_update_yfinance_data(symbol: str):
    """
    Check if the historical data for the symbol exists.
    If it exists, only fetch the most recent data and update the file.
    If it doesn't exist, fetch the full 2 years of data.
    """
    data_dir = "data/yfinance"
    date_today = datetime.now().strftime("%Y-%m-%d")
    file_path = f"{data_dir}/{symbol}_yf.csv"

    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, load it and check the last date
        existing_data = pd.read_csv(file_path, index_col=0)
        existing_data.index = pd.to_datetime(existing_data.index, errors='coerce')
        last_date = existing_data.index[-1]

        # If the last date is today, no need to fetch new data
        if last_date.date() == datetime.today().date():
            logger.info(f"{symbol} already has today's data, skipping fetch.")
            return

        # If the last date is not today, fetch new data from the last date
        start_date = last_date + timedelta(days=1)
        logger.info(f"Fetching new data for {symbol} from {start_date.date()} to {date_today}")
        new_data = await fetch_yfinance_data(symbol, start_date=start_date, end_date=datetime.today())
        updated_data = pd.concat([existing_data, new_data])

    else:
        # If the file doesn't exist, fetch full data for 2 years
        logger.info(f"{symbol} data not found, fetching full 2 years of data.")
        new_data = await fetch_yfinance_data(symbol)
        updated_data = new_data

    # Save the updated data back to the CSV
    updated_data.to_csv(file_path)
    logger.info(f"Data for {symbol} saved to {file_path}")


async def main():
    try:
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Fetch new sentiment data
        scraper = NewsScraper()
        await scraper.fetch_news()

        # Fetch current day's historical candles data
        symbols = config['assets']
        await fetch_candles_data(symbols)

        # Fetch and update yfinance data for all assets
        for symbol in symbols:
            await fetch_and_update_yfinance_data(symbol)

        # Preprocess new sentiment data
        sentiment_file = "data/sentiment_data.csv"
        for symbol in symbols:
            preprocess_data(symbol, sentiment_file, sequence_length=config['model_params']['lstm']['sequence_length'])

        # Train model with new sentiment data
        for symbol in symbols:
            predictions = train_model(symbol)

            # Safely check if predictions are a float or array
            if isinstance(predictions, (np.ndarray, list)) and len(predictions) > 0:
                last_prediction = predictions[-1]
                logger.info(f"Generated signal for {symbol}: {'BUY' if last_prediction > 0 else 'SELL'} (prediction: {last_prediction:.4f})")
            elif isinstance(predictions, (float, int)):
                last_prediction = predictions
                logger.info(f"Generated signal for {symbol}: {'BUY' if last_prediction > 0 else 'SELL'} (prediction: {last_prediction:.4f})")
            else:
                logger.warning(f"No valid predictions generated for {symbol}")

        # Run live trading
        trader = MetaApiTrader(config)
        await trader.run()

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
    finally:
        logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())

