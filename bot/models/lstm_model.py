# models/lstm_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os

class LSTMModel:
    def __init__(self, data_folder="data", model_folder="models", lookback=50):
        self.data_folder = data_folder
        self.model_folder = model_folder
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, asset):
        """Load preprocessed data for the asset and prepare it for LSTM."""
        file_path = os.path.join(self.data_folder, f"{asset}_processed.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed data not found for {asset}")

        df = pd.read_csv(file_path)
        return df

    def prepare_data(self, df, feature='close_normalized'):
        """Prepare data for LSTM model."""
        data = df[feature].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def build_model(self):
        """Build and compile the LSTM model."""
        model = Sequential()

        # LSTM layers with Dropout for regularization
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.lookback, 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(units=25))
        model.add(Dense(units=1))  # Predicting the future price

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, asset, epochs=10, batch_size=32):
        """Train the LSTM model on the asset's data."""
        df = self.load_data(asset)
        X, y = self.prepare_data(df)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Build the LSTM model
        model = self.build_model()

        # Train the model
        print(f"Training LSTM model for {asset}...")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

        # Save the trained model
        model_path = os.path.join(self.model_folder, f"lstm_{asset}.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")

        return model, X_test, y_test

    def predict(self, model, X_test):
        """Make predictions using the trained model."""
        predictions = model.predict(X_test)
        return self.scaler.inverse_transform(predictions)

# Example usage:
if __name__ == "__main__":
    lstm_model = LSTMModel()

    # Train model for EURUSD
    trained_model, X_test, y_test = lstm_model.train_model('EURUSD')

    # Make predictions
    predictions = lstm_model.predict(trained_model, X_test)

    print("Predictions: ", predictions[:5])

