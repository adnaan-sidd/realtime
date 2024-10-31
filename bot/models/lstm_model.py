import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import pickle
import json
import logging

logger = logging.getLogger(__name__)

def create_lstm_model(input_shape: tuple, units: list = [128, 64, 32], dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Create LSTM model architecture.
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    
    # Middle LSTM layers
    for unit in units[1:-1]:
        model.add(LSTM(unit, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    # Final LSTM layer
    model.add(LSTM(units[-1]))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    return model

def train_model(symbol: str, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2, early_stopping_patience: int = 10):
    """
    Train LSTM model and return the latest prediction.
    Returns:
        float: The latest prediction for the symbol
    """
    try:
        model_path = f'models/{symbol}_best_model.keras'
        
        # Load or train the model
        if os.path.exists(model_path):
            logger.info(f"Model for {symbol} already trained. Loading the model from {model_path}.")
            model = tf.keras.models.load_model(model_path)
        else:
            # Load preprocessed data
            X = np.load(f'data/{symbol}_X.npy')
            y = np.load(f'data/{symbol}_y.npy')
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Create and train model
            model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            ]
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save training history
            with open(f'models/{symbol}_training_history.json', 'w') as f:
                json.dump(history.history, f)

        # Get latest prediction
        return get_latest_prediction(symbol, model)

    except Exception as e:
        logger.error(f"Error in train_model for {symbol}: {e}")
        return None

def get_latest_prediction(symbol: str, model=None):
    """
    Get the latest prediction for a symbol.
    """
    try:
        # Load test data and scalers
        X = np.load(f'data/{symbol}_X.npy')
        with open(f'data/{symbol}_scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        close_scaler = scalers['Close']
        
        # Get the most recent data point
        latest_data = X[-1:]  # Take the last sequence
        
        # Make prediction
        prediction = model.predict(latest_data, verbose=0)
        
        # Inverse transform the prediction
        prediction_unscaled = close_scaler.inverse_transform(prediction)
        
        # Return the single prediction value
        return float(prediction_unscaled[0][0])

    except Exception as e:
        logger.error(f"Error getting latest prediction for {symbol}: {e}")
        return None

def make_predictions(symbol: str):
    """
    Make predictions using the trained model.
    """
    try:
        model_path = f'models/{symbol}_best_model.keras'
        if not os.path.exists(model_path):
            logger.warning(f"No model found for {symbol}. Please train the model first.")
            return None

        model = tf.keras.models.load_model(model_path)
        return get_latest_prediction(symbol, model)

    except Exception as e:
        logger.error(f"Error in make_predictions for {symbol}: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os.makedirs('models', exist_ok=True)
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    
    for symbol in symbols:
        prediction = train_model(symbol)
        if prediction is not None:
            logger.info(f"Latest prediction for {symbol}: {prediction}")
        else:
            logger.warning(f"Could not get prediction for {symbol}")