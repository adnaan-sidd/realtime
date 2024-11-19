import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split
import os
import pickle
import json
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_lstm_model(input_shape: tuple, units: list = [128, 64, 32], dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Create LSTM model architecture.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, features)
        units (list): List of units for each LSTM layer
        dropout_rate (float): Dropout rate between layers
        
    Returns:
        tf.keras.Model: Compiled LSTM model
    """
    model = Sequential()

    model.add(Bidirectional(LSTM(units[0], return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(dropout_rate))

    for unit in units[1:-1]:
        model.add(Bidirectional(LSTM(unit, return_sequences=True)))
        model.add(Dropout(dropout_rate))

    model.add(Bidirectional(LSTM(units[-1])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer
    
    return model

def load_data(symbol: str):
    """
    Load preprocessed data for the given symbol.
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        tuple: Input features (X) and target values (y)
    """
    try:
        X = np.load(f'data/{symbol}_X.npy')
        y = np.load(f'data/{symbol}_y.npy')
        return X, y
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        raise

def train_model(symbol: str, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2, 
                early_stopping_patience: int = 20):
    """
    Train LSTM model and return the latest prediction.
    
    Args:
        symbol (str): Trading symbol (e.g., 'EURUSD')
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        validation_split (float): Fraction of data to use for validation
        early_stopping_patience (int): Number of epochs to wait before early stopping
        
    Returns:
        float: The latest prediction for the symbol
    """
    model_path = f'models/{symbol}_best_model.keras'
    log_dir = f"logs/{symbol}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Load or train the model
    if os.path.exists(model_path):
        logger.info(f"Model for {symbol} already trained. Loading from {model_path}.")
        model = tf.keras.models.load_model(model_path)
    else:
        X, y = load_data(symbol)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            tensorboard_callback
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
            json.dump({key: [float(val) for val in values] for key, values in history.history.items()}, f)

    return get_latest_prediction(symbol, model)

def get_latest_prediction(symbol: str, model: tf.keras.Model) -> float:
    """
    Get the latest prediction for a symbol using the trained model.
    
    Args:
        symbol (str): Trading symbol
        model (tf.keras.Model): Pre-loaded model
        
    Returns:
        float: Predicted price value
    """
    try:
        X = np.load(f'data/{symbol}_X.npy')
        with open(f'data/{symbol}_scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        close_scaler = scalers['Close']

        latest_data = X[-1:]  # Take the last sequence
        prediction = model.predict(latest_data, verbose=0)

        # Inverse transform the prediction
        prediction_unscaled = close_scaler.inverse_transform(prediction)
        return float(prediction_unscaled[0][0])

    except Exception as e:
        logger.error(f"Error getting latest prediction for {symbol}: {e}")
        return None

def make_predictions(symbol: str) -> tuple:
    """
    Make predictions using the trained model for the given trading symbol.
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        tuple: (predicted price value, duration in seconds)
    """
    model_path = f'models/{symbol}_best_model.keras'

    if not os.path.exists(model_path):
        logger.warning(f"No model found for {symbol}. Please train the model first.")
        return None, None  # Return None for both values if there is no model

    model = tf.keras.models.load_model(model_path)
    prediction = get_latest_prediction(symbol, model)

    # Assuming 'duration' is a constant, define it as per your requirements
    duration = 3600  # Example duration of 1 hour; adjust if needed

    return prediction, duration

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

    for symbol in symbols:
        prediction = train_model(symbol)
        if prediction is not None:
            logger.info(f"Latest prediction for {symbol}: {prediction}")
        else:
            logger.warning(f"Could not get prediction for {symbol}")