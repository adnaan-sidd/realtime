import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import pickle
import json

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
    Train LSTM model using preprocessed data or load an existing model if already trained.
    """
    model_path = f'models/{symbol}_best_model.keras'

    # Check if the model already exists and load it to avoid retraining
    if os.path.exists(model_path):
        print(f"Model for {symbol} already trained. Loading the model from {model_path}.")
        model = tf.keras.models.load_model(model_path)
        return model

    # Load preprocessed data
    X = np.load(f'data/{symbol}_X.npy')
    y = np.load(f'data/{symbol}_y.npy')
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Create model
    model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)  # Save model with .keras extension
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss for {symbol}: {test_loss:.4f}")
    print(f"Test MAE for {symbol}: {test_mae:.4f}")
    
    # Save training history
    with open(f'models/{symbol}_training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    return model

def make_predictions(symbol: str):
    """
    Make predictions using the trained model.
    """
    model_path = f'models/{symbol}_best_model.keras'
    if not os.path.exists(model_path):
        print(f"No model found for {symbol}. Please train the model first.")
        return

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load test data and scalers
    X = np.load(f'data/{symbol}_X.npy')
    with open(f'data/{symbol}_scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    close_scaler = scalers['Close']
    
    # Use last 20% of data for testing
    _, X_test, _, _ = train_test_split(X, X, test_size=0.2, shuffle=False)
    
    # Predict and inverse-transform
    predictions = model.predict(X_test)
    predictions_unscaled = close_scaler.inverse_transform(predictions)
    
    return predictions_unscaled

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    # Assets to train and predict
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    
    for symbol in symbols:
        # Train the model or load the existing one
        model = train_model(symbol)
        
        # Make predictions with the trained model
        predictions = make_predictions(symbol)
        
        if predictions is not None:
            # Save predictions for later analysis or use
            np.save(f'data/{symbol}_predictions.npy', predictions)
