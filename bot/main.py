# main.py
import sys
import os
import asyncio
import yaml
import logging
import numpy as np
from datetime import datetime

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Add directories to the Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for directory in ['news', 'models', 'strategies', 'indicators']:
    sys.path.append(os.path.join(CURRENT_DIR, directory))

# Import after path setup
from news_scraper import NewsScraper
from preprocess_data import preprocess_data
from lstm_model import train_model
from metaapi_trader import MetaApiTrader

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['logs', 'data', 'models', 'config']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_config():
    """Load configuration from YAML file."""
    try:
        config_path = os.path.join(CURRENT_DIR, 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

async def fetch_and_process_news(scraper, symbols, sentiment_file, sequence_length):
    """Fetch news and process data for all symbols."""
    try:
        logger.info("Fetching news data...")
        await scraper.fetch_news()
        
        logger.info("Processing data for all symbols...")
        for symbol in symbols:
            try:
                preprocess_data(
                    symbol, 
                    sentiment_file, 
                    sequence_length=sequence_length
                )
                logger.info(f"Preprocessed data for {symbol}")
            except Exception as e:
                logger.error(f"Error preprocessing data for {symbol}: {e}")
                
    except Exception as e:
        logger.error(f"Error in fetch_and_process_news: {e}")
        raise

async def generate_trading_signals(symbols, config):
    """Generate trading signals for all symbols."""
    trading_signals = {}
    threshold = config.get('prediction_threshold', 0.0)
    
    for symbol in symbols:
        try:
            # Get model predictions
            prediction = train_model(symbol)
            
            if prediction is not None:
                # Determine trading signal based on prediction
                signal = {
                    'symbol': symbol,
                    'prediction': float(prediction),  # Convert numpy float to Python float
                    'action': 'buy' if prediction > threshold else 'sell',
                    'confidence': float(abs(prediction - threshold) / threshold if threshold != 0 else abs(prediction))
                }
                trading_signals[symbol] = signal
                logger.info(f"Generated signal for {symbol}: {signal['action'].upper()} (prediction: {prediction:.4f})")
            else:
                logger.warning(f"No valid prediction generated for {symbol}")
                
        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            continue
            
    return trading_signals

class TradingManager:
    def __init__(self, config):
        self.trader = MetaApiTrader(config)
        self.config = config

    async def process_signals(self, trading_signals):
        """Process trading signals and execute trades"""
        try:
            if not trading_signals:
                logger.warning("No valid trading signals to process")
                return

            # Run the trader with the trading signals
            await self.trader.run(trading_signals)
            
        except Exception as e:
            logger.error(f"Error processing trading signals: {e}")
            raise

async def main():
    """Main execution function."""
    try:
        # Setup
        setup_directories()
        config = load_config()
        
        # Initialize components
        scraper = NewsScraper()
        symbols = config['assets']
        sentiment_file = os.path.join(CURRENT_DIR, "data", "sentiment_data.csv")
        sequence_length = config['model_params']['lstm']['sequence_length']
        
        # Main workflow
        logger.info("Starting trading system...")
        
        # Step 1: Fetch and process news
        await fetch_and_process_news(scraper, symbols, sentiment_file, sequence_length)
        
        # Step 2: Generate trading signals
        trading_signals = await generate_trading_signals(symbols, config)
        
        # Step 3: Execute trading with the trading manager
        trading_manager = TradingManager(config)
        await trading_manager.process_signals(trading_signals)
        
        logger.info("Trading cycle completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise
    finally:
        logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Trading system stopped by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
