import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import os
import traceback

class LSTMModel:
    def __init__(self, data_folder="data", model_folder="models", max_lookback=50, min_lookback=5):
        self.data_folder = data_folder
        self.model_folder = model_folder
        self.max_lookback = max_lookback
        self.min_lookback = min_lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, asset):
        """Load preprocessed data for the asset and prepare it for LSTM."""
        file_path = os.path.join(self.data_folder, f"{asset}_processed.csv")
        abs_file_path = os.path.abspath(file_path)
        print(f"Attempting to load file: {abs_file_path}")
        
        if not os.path.exists(abs_file_path):
            print(f"Current working directory: {os.getcwd()}")
            print(f"Contents of {os.path.dirname(abs_file_path)}:")
            print(os.listdir(os.path.dirname(abs_file_path)))
            raise FileNotFoundError(f"Processed data not found for {asset} at {abs_file_path}")

        df = pd.read_csv(abs_file_path)
        print(f"Data loaded for {asset}. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df

    def prepare_data(self, df, feature='close_normalized'):
        """Prepare data for LSTM model with dynamic lookback period."""
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        
        data = df[feature].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)

        # Dynamically adjust lookback period
        lookback = min(self.max_lookback, max(self.min_lookback, len(scaled_data) // 10))
        print(f"Using lookback period of {lookback}")

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i, 0])
            y.append(scaled_data[i, 0])

        if len(X) == 0:
            raise ValueError(f"Not enough data to create sequences. Data length: {len(scaled_data)}, Minimum required: {self.min_lookback + 1}")

        X, y = np.array(X), np.array(y)
        print(f"Prepared data shapes before reshaping - X: {X.shape}, y: {y.shape}")
        
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        print(f"Prepared data shapes after reshaping - X: {X.shape}, y: {y.shape}")
        return X, y, lookback

    def build_model(self, lookback):
        """Build and compile the LSTM model."""
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, asset, epochs=10, batch_size=32):
        """Train the LSTM model on the asset's data."""
        df = self.load_data(asset)
        X, y, lookback = self.prepare_data(df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = self.build_model(lookback)
        print(f"Training LSTM model for {asset}...")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

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
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']

    for asset in assets:
        try:
            print(f"\nProcessing {asset}...")
            trained_model, X_test, y_test = lstm_model.train_model(asset)
            predictions = lstm_model.predict(trained_model, X_test)
            print(f"{asset} Predictions (first 5):")
            print(predictions[:5])

        except FileNotFoundError as e:
            print(f"Error for {asset}: {e}")
            print(f"Please ensure that the processed data file for {asset} exists in the correct location.")
        except ValueError as e:
            print(f"Error preparing data for {asset}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for {asset}:")
            print(traceback.format_exc())  # This will print the full traceback

    print("\nAll assets processed.")
