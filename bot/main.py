import sys
import os
import asyncio
import yaml
import logging
import numpy as np

# Add directories to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'news')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'strategies')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'indicators')))

from news_scraper import NewsScraper
from preprocess_data import preprocess_data
from lstm_model import train_model
from metaapi_trader import MetaApiTrader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    try:
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Fetch new sentiment data
        scraper = NewsScraper()
        await scraper.fetch_news()

        # Preprocess new sentiment data
        symbols = config['assets']
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
