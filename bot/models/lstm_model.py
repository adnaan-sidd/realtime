import os
import tensorflow as tf
import logging
import warnings

# Disable OneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set TensorFlow logging level to suppress info logs
tf.get_logger().setLevel('ERROR')

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress TensorFlow related warning logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import os
import pickle
import json
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_hybrid_model(input_shape: tuple, units: list = [128, 64, 32], dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Create a hybrid CNN-LSTM model architecture.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, features)
        units (list): List of units for each LSTM layer
        dropout_rate (float): Dropout rate between layers
        
    Returns:
        tf.keras.Model: Compiled hybrid CNN-LSTM model
    """
    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)

    # LSTM layers
    x = Bidirectional(LSTM(units[0], return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)

    for unit in units[1:-1]:
        x = Bidirectional(LSTM(unit, return_sequences=True))(x)
        x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(units[-1], return_sequences=True))(x)

    # Attention layer
    attention = Attention()([x, x])
    x = Dropout(dropout_rate)(attention)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)  # Output layer

    model = Model(inputs, outputs)
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
    Train hybrid CNN-LSTM model and return the latest prediction.
    
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
        try:
            X, y = load_data(symbol)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = create_hybrid_model(input_shape=(X.shape[1], X.shape[2]))
            model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                tensorboard_callback,
                LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 20))
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
        except Exception as e:
            logger.error(f"Error during training for {symbol}: {e}")
            return None

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
        close_scaler = scalers.get('Close')

        if close_scaler is None:
            logger.error(f"Scaler for 'Close' not found for {symbol}.")
            return None

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
        tuple: (predicted price values, duration in seconds)
    """
    model_path = f'models/{symbol}_best_model.keras'

    if not os.path.exists(model_path):
        logger.warning(f"No model found for {symbol}. Please train the model first.")
        return None, None  # Return None for both values if there is no model

    model = tf.keras.models.load_model(model_path)
    X = np.load(f'data/{symbol}_X.npy')
    with open(f'data/{symbol}_scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    close_scaler = scalers.get('Close')

    if close_scaler is None:
        logger.error(f"Scaler for 'Close' not found for {symbol}.")
        return None, None

    predictions = model.predict(X, verbose=0)
    predictions_unscaled = close_scaler.inverse_transform(predictions)
    return predictions_unscaled.flatten(), 3600  # Example duration of 1 hour; adjust if needed

def update_model_with_new_data(symbol: str, new_X: np.ndarray, new_y: np.ndarray):
    """
    Update the model with new data.
    
    Args:
        symbol (str): Trading symbol
        new_X (np.ndarray): New input features
        new_y (np.ndarray): New target values
    """
    try:
        model_path = f'models/{symbol}_best_model.keras'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            model.fit(new_X, new_y, epochs=10, batch_size=32, verbose=1)  # Adjust parameters as needed
            model.save(model_path)
        else:
            logger.error(f"Model for {symbol} not found. Please train the model first.")
    except Exception as e:
        logger.error(f"Error updating model for {symbol} with new data: {e}")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

    for symbol in symbols:
        prediction = train_model(symbol)
        if prediction is not None:
            logger.info(f"Latest prediction for {symbol}: {prediction}")
        else:
            logger.warning(f"Could not get prediction for {symbol}")
