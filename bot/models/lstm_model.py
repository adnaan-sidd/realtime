import os
import numpy as np
import tensorflow as tf
import logging
import json
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical

# Suppress TensorFlow and other warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Fixed the typo here

def load_config(config_path: str):
    with open(config_path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

def create_hybrid_model(input_shape: tuple, config) -> tf.keras.Model:
    # Input layer
    inputs = Input(shape=input_shape)

    # Accessing convolution parameters from config
    conv_filters = config['conv_params']['conv_filters']
    kernel_size = config['conv_params']['kernel_size']
    pool_size = 2

    # 1D Convolutional Layer followed by Max Pooling
    x = Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(config['model_params']['lstm']['dropout'])(x)

    # LSTM layers
    for unit in config['model_params']['lstm']['layers']:
        x = Bidirectional(LSTM(unit, return_sequences=True))(x)
        x = Dropout(config['model_params']['lstm']['dropout'])(x)

    # Flatten the output from LSTM layers
    x = Flatten()(x)

    # Use the num_classes from config
    outputs = Dense(config['model_params']['num_classes'], activation='softmax')(x)

    # Create the model
    model = Model(inputs, outputs)
    
    # Prepare the optimizer based on the configuration
    optimizer_type = config['model_params']['lstm']['optimizer']['type']
    learning_rate = config['model_params']['lstm']['optimizer'].get('learning_rate', 0.001)

    # Initialize the optimizer
    if optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer type '{optimizer_type}' is not recognized.")

    # Compile the model
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def load_data(symbol: str):
    try:
        # Load processed input data and labels
        data_directory = "data"
        X = np.load(f'{data_directory}/{symbol}_X.npy')
        y = np.load(f'{data_directory}/{symbol}_y.npy')
        logger.info(f"Loaded data for {symbol}: X shape {X.shape}, y shape {y.shape}.")
        return X, y
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {e}")
        raise

def check_for_new_data(symbol: str) -> bool:
    existing_data_path = f'data/{symbol}_X.npy'
    new_data_path = f'data/{symbol}_X_new.npy'
    
    try:
        if os.path.exists(existing_data_path) and os.path.exists(new_data_path):
            existing_data = np.load(existing_data_path)
            new_data = np.load(new_data_path)
            return existing_data.shape[0] < new_data.shape[0]  # If new data has more rows
        else:
            logger.info("No existing or new data found.")
            return False
    except Exception as e:
        logger.error(f"Error checking for new data for {symbol}: {e}")
        return False

def train_model(symbol: str, config, model_path: str = None) -> str:
    model_path = model_path or f'models/{symbol}_best_model.keras'
    log_dir = f"{config['system']['log_directory']}/{symbol}"

    # Check for existing predictions
    if os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}.")
        model = tf.keras.models.load_model(model_path)

        # Check for new data
        if check_for_new_data(symbol):
            logger.info("New data detected. Retraining the model...")
            existing_X, existing_y = load_data(symbol)
            new_X, new_y = load_data(symbol)  # Load new data, assuming it has been processed
            all_X = np.concatenate((existing_X, new_X))
            all_y = np.concatenate((existing_y, new_y))
        else:
            logger.info("No new data detected. Continuing with existing model.")
            all_X, all_y = load_data(symbol)
    else:
        logger.info("No existing model found. Training from scratch.")
        
        # Load data for training
        all_X, all_y = load_data(symbol)
        
        # Create a new model since there was no existing model
        input_shape = all_X.shape[1:]  # Assuming shape is (samples, time_steps, features)
        model = create_hybrid_model(input_shape, config)

    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(all_X, all_y, test_size=config['preprocessing']['train_test_split'], shuffle=False)

    # Convert y_train and y_val to one-hot encoding
    y_train_one_hot = to_categorical(y_train, num_classes=config['model_params']['num_classes'])
    y_val_one_hot = to_categorical(y_val, num_classes=config['model_params']['num_classes'])

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config['model_params']['lstm']['epochs'] // 10, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=config['model_params']['lstm']['epochs'] // 20, min_lr=1e-6),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    # Fit the model
    history = model.fit(
        X_train, 
        y_train_one_hot,
        epochs=config['model_params']['lstm']['epochs'],
        batch_size=config['model_params']['lstm']['batch_size'],
        validation_data=(X_val, y_val_one_hot),
        callbacks=callbacks,
        verbose=1
    )

    save_training_history(symbol, history)
    return get_latest_prediction(symbol, model)

def save_training_history(symbol: str, history):
    with open(f'models/{symbol}_training_history.json', 'w') as f:
        json.dump({key: [float(val) for val in values] for key, values in history.history.items()}, f)

def get_latest_prediction(symbol: str, model: tf.keras.Model) -> str:
    try:
        data_directory = "data"
        X = np.load(f'{data_directory}/{symbol}_X.npy')
        latest_data = X[-1:]  # Take the last sequence
        predictions = model.predict(latest_data, verbose=0)

        predicted_class = np.argmax(predictions, axis=1)[0]
        signal_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}  # Map indices to signals
        return signal_map[predicted_class]

    except Exception as e:
        logger.error(f"Error getting latest prediction for {symbol}: {e}")
        return None

def make_predictions(symbol: str) -> str:
    model_path = f'models/{symbol}_best_model.keras'

    if not os.path.exists(model_path):
        logger.warning(f"No model found for {symbol}. Please train the model first.")
        return None 

    model = tf.keras.models.load_model(model_path)
    data_directory = "data"
    X = np.load(f'{data_directory}/{symbol}_X.npy')

    predictions = model.predict(X, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    # Map to signals
    signal_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
    predicted_labels = [signal_map[c] for c in predicted_classes]

    return predicted_labels[-1]  # Return the last predicted label

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Load configurations
    config = load_config('config/config.yaml')

    symbols = config['assets']

    for symbol in symbols:
        prediction = train_model(symbol, config)
        if prediction is not None:
            logger.info(f"Latest prediction for {symbol}: {prediction}")
        else:
            logger.warning(f"Could not get prediction for {symbol}")