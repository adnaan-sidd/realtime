import os
import numpy as np
import tensorflow as tf
import logging
import warnings
import pickle
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler

# Suppress TensorFlow and other warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

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
    for unit in units:
        x = Bidirectional(LSTM(unit, return_sequences=True))(x)
        x = Dropout(dropout_rate)(x)

    # Attention layer
    attention = Attention()([x, x])
    x = Flatten()(attention)
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

def train_model(symbol: str, epochs: int = 10, batch_size: int = 32, validation_split: float = 0.2, 
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

    # Always load new data for training
    try:
        new_X, new_y = load_data(symbol)

        # If the model exists, load it and get existing data
        if os.path.exists(model_path):
            logger.info(f"Model for {symbol} already exists. Loading from {model_path}.")
            model = tf.keras.models.load_model(model_path)

            # Load existing training data; create new if not present
            existing_X = np.load(f'data/{symbol}_X_existing.npy')
            existing_y = np.load(f'data/{symbol}_y_existing.npy')

            # Combine existing data with new data
            combined_X = np.concatenate((existing_X, new_X), axis=0)
            combined_y = np.concatenate((existing_y, new_y), axis=0)

            # Save combined data for the next training
            np.save(f'data/{symbol}_X_existing.npy', combined_X)
            np.save(f'data/{symbol}_y_existing.npy', combined_y)

        else:
            logger.info(f"Creating a new model for {symbol}.")
            model = create_hybrid_model(input_shape=(new_X.shape[1], new_X.shape[2]))
            combined_X, combined_y = new_X, new_y  # On first train, this is the data to use

            # Initialize existing data files
            logger.info(f"Creating initial empty data files for {symbol}.")
            np.save(f'data/{symbol}_X_existing.npy', np.empty((0, new_X.shape[1], new_X.shape[2])))  # Shape for 3D array
            np.save(f'data/{symbol}_y_existing.npy', np.empty((0,)))  # Assuming y is 1-D

        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            tensorboard_callback,
            LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (-epoch / 20))
        ]

        # Split the combined dataset for training and validation
        X_train, X_val, y_train, y_val = train_test_split(combined_X, combined_y, test_size=validation_split, shuffle=False)

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        save_training_history(symbol, history)

    except Exception as e:
        logger.error(f"Error during training for {symbol}: {e}")
        return None

    return get_latest_prediction(symbol, model)

def save_training_history(symbol: str, history):
    """
    Save the training history of the model to a JSON file.
    
    Args:
        symbol (str): Trading symbol
        history: Training history object
    """
    with open(f'models/{symbol}_training_history.json', 'w') as f:
        json.dump({key: [float(val) for val in values] for key, values in history.history.items()}, f)

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

def load_scalers(symbol: str) -> dict:
    """
    Load scalers for the given symbol.
    
    Args:
        symbol (str): Trading symbol
        
    Returns:
        dict: Scalers associated with the symbol
    """
    try:
        with open(f'data/{symbol}_scalers.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading scalers for {symbol}: {e}")
        raise

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

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)  # Ensure the data directory exists

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

    for symbol in symbols:
        # Initialize existing data files if they do not exist
        if not os.path.exists(f'data/{symbol}_X_existing.npy'):
            logger.info(f"Creating initial empty data files for {symbol}.")
            # Update expected_shape to match the one used in preprocess_data.py
            expected_sequence_length = 60  # This should match the sequence length used during preprocessing
            expected_features = len(['Open', 'High', 'Low', 'Close', 'Volume', 'positive', 'negative', 'neutral', 'Returns', 'RSI', 'Momentum', 'ROC'])  # Count your feature columns
            np.save(f'data/{symbol}_X_existing.npy', np.empty((0, expected_sequence_length, expected_features)))  # Shape for 3D array
            np.save(f'data/{symbol}_y_existing.npy', np.empty((0,)))  # Assuming y is 1-D

        prediction = train_model(symbol)
        if prediction is not None:
            logger.info(f"Latest prediction for {symbol}: {prediction}")
        else:
            logger.warning(f"Could not get prediction for {symbol}")