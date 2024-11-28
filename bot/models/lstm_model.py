import os
import numpy as np
import tensorflow as tf
import logging
import warnings
import pickle
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# Suppress TensorFlow and other warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_hybrid_model(input_shape: tuple, units: list = [128, 64], dropout_rate: float = 0.2) -> tf.keras.Model:
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
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)

    # LSTM layers
    for unit in units:
        x = Bidirectional(LSTM(unit, return_sequences=True))(x)
        x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    outputs = Dense(1)(x)  # Output layer

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
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

def train_model(
    symbol: str, 
    epochs: int = 10, 
    batch_size: int = 32, 
    validation_split: float = 0.2, 
    early_stopping_patience: int = 20
):
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

    # Load new data for training
    new_X, new_y = load_data(symbol)

    # Check if the model exists and if the data is the same
    if os.path.exists(model_path):
        logger.info(f"Model for {symbol} already exists. Loading from {model_path}.")
        model = tf.keras.models.load_model(model_path)

        # Check if there's new data by comparing with the existing data on disk
        existing_X = np.load(f'data/{symbol}_X_existing.npy')
        existing_y = np.load(f'data/{symbol}_y_existing.npy')

        if np.array_equal(new_X, existing_X) and np.array_equal(new_y, existing_y):
            logger.info(f"No new data available for {symbol}. Skipping training.")
            return get_latest_prediction(symbol, model)

    # Proceed with training if model does not exist or if new data is present
    logger.info(f"Training model for {symbol}.")
    if not os.path.exists(model_path):
        model = create_hybrid_model(input_shape=(new_X.shape[1], new_X.shape[2]))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        tensorboard_callback
    ]

    # Split dataset for training and validation
    X_train, X_val, y_train, y_val = train_test_split(new_X, new_y, test_size=validation_split, shuffle=False)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    save_training_history(symbol, history)
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
            # Update the sequence length and features according to your preprocessing
            expected_sequence_length = 60  # This should match the sequence length used during preprocessing
            expected_features = 11  # Adjust this based on the number of features you are using
            np.save(f'data/{symbol}_X_existing.npy', np.empty((0, expected_sequence_length, expected_features)))  # Shape for 3D array
            np.save(f'data/{symbol}_y_existing.npy', np.empty((0,)))  # Assuming y is 1-D

        prediction = train_model(symbol)
        if prediction is not None:
            logger.info(f"Latest prediction for {symbol}: {prediction}")
        else:
            logger.warning(f"Could not get prediction for {symbol}")