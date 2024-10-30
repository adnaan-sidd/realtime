import sys
import os
import asyncio
import yaml  # For configuration loading

# Add directories to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'news')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'strategies')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'indicators')))

from news_scraper import NewsScraper
from preprocess_data import preprocess_data
from lstm_model import train_model
from backtest import Backtest
from metaapi_trader import MetaApiTrader

async def main():
    # Load configuration from a YAML file
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Fetch new sentiment data
    scraper = NewsScraper()
    await scraper.fetch_news()  # Ensure this is an async call if needed

    # Preprocess new sentiment data
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
    sentiment_file = "data/sentiment_data.csv"
    for symbol in symbols:
        preprocess_data(symbol, sentiment_file, sequence_length=config['sequence_length'])

    # Train model with new sentiment data
    for symbol in symbols:
        train_model(symbol)

    # Live trading
    trader = MetaApiTrader(config)
    await trader.run()

if __name__ == "__main__":
    asyncio.run(main())
