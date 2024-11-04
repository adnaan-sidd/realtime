import sys
import os
import asyncio
import yaml
import logging
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Add directories to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'news')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'strategies')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'indicators')))

from news_scraper import NewsScraper
from preprocess_data import preprocess_data
from lstm_model import train_model
from metaapi_trader import MetaApiTrader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def validate_prediction(symbol, prediction, trader):
    """Validate prediction using technical indicators and market conditions."""
    try:
        connection = trader.account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized()
        
        # Get historical data for technical analysis
        candles = await connection.get_candles(symbol, '1h', 100)
        df = pd.DataFrame([{
            'open': candle['open'],
            'high': candle['high'],
            'low': candle['low'],
            'close': candle['close'],
            'volume': candle['tickVolume']
        } for candle in candles])
        
        # Calculate technical indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['close'], 14)
        
        # Get current market conditions
        current_price = await connection.get_symbol_price(symbol)
        current_close = current_price['ask']
        
        # Validation rules for enhanced accuracy
        sma_trend = df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]
        rsi_condition = 30 <= df['RSI'].iloc[-1] <= 70
        price_trend = current_close > df['SMA_20'].iloc[-1]
        
        # Combine LSTM prediction with technical analysis
        if prediction > 0:  # Original prediction is BUY
            return prediction if (sma_trend and price_trend and rsi_condition) else 0
        else:  # Original prediction is SELL
            return prediction if (not sma_trend and not price_trend and rsi_condition) else 0
            
    except Exception as e:
        logger.error(f"Error validating prediction for {symbol}: {e}")
        return prediction

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

async def check_market_conditions():
    """Check if current market conditions are suitable for trading."""
    now = datetime.utcnow()
    
    # Check market hours (Forex market runs 24/5)
    if now.weekday() >= 5:  # Weekend
        return False
        
    # Avoid trading during high-impact news events
    # You would need to implement news event checking here
    
    return True

async def main():
    try:
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        trader = MetaApiTrader(config)
        await trader.connect_account()

        while True:  # Continuous loop
            if not await check_market_conditions():
                logger.info("Market conditions not suitable for trading. Waiting...")
                await asyncio.sleep(300)  # Wait 5 minutes
                continue

            # Fetch new sentiment data
            scraper = NewsScraper()
            await scraper.fetch_news()

            # Process symbols
            symbols = config['assets']
            sentiment_file = "data/sentiment_data.csv"
            
            for symbol in symbols:
                try:
                    # Preprocess data with enhanced features
                    preprocess_data(symbol, sentiment_file, 
                                  sequence_length=config['model_params']['lstm']['sequence_length'])
                    
                    # Get predictions
                    predictions = train_model(symbol)
                    
                    if isinstance(predictions, (np.ndarray, list)) and len(predictions) > 0:
                        last_prediction = predictions[-1]
                        # Validate prediction
                        validated_prediction = await validate_prediction(symbol, last_prediction, trader)
                        
                        if abs(validated_prediction) > config['model_params']['confidence_threshold']:
                            logger.info(f"High confidence signal for {symbol}: "
                                      f"{'BUY' if validated_prediction > 0 else 'SELL'} "
                                      f"(prediction: {validated_prediction:.4f})")
                            
                            # Execute trade through MetaApiTrader
                            await trader.execute_trade(symbol, 
                                                     "ORDER_TYPE_BUY" if validated_prediction > 0 else "ORDER_TYPE_SELL",
                                                     validated_prediction)
                        else:
                            logger.info(f"Low confidence signal for {symbol}, no trade executed")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Wait before next analysis cycle
            await asyncio.sleep(config.get('analysis_interval', 1800))  # Default 30 minutes

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
    finally:
        logger.info("Trading system shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
