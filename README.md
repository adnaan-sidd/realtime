# Trading Bot

This trading bot leverages AI to make automated trades on financial assets using historical data, sentiment analysis from news articles, and a predictive model based on LSTM. It is built to operate with the MetaTrader 5 platform and uses multiple APIs to fetch data and analyze sentiment.

## Features

- **Predictive Model**: Utilizes an LSTM model to predict asset price movements.
- **Sentiment Analysis**: Analyzes news sentiment with a transformer model to influence trading decisions.
- **Data Sources**: Fetches data from Yahoo Finance and MetaTrader 5 (MT5) for asset price history and Bing API for news data.
- **Automated Trading**: Sends trade signals directly to MT5 for assets such as EURUSD, GBPUSD, USDJPY, and XAUUSD.
- **Flexible Configuration**: Fully configurable via `config.yaml` file for assets, trading parameters, and API keys.

## Installation

### Prerequisites

- **Python 3.10.0rc2**
- **MetaTrader 5** installed and configured
- **API Keys** for Bing News API, MetaAPI (if needed), and any other external APIs
- **Virtual Environment** (recommended)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/adnaan-sidd/realtime.git
   cd your-repo-name
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Install the required packages from `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure `config.yaml`**

   Update the `config/config.yaml` file with your specific settings:

   - **API Keys** for Bing News and MT5.
   - **Assets** to be traded.
   - **News and Model Parameters** such as refresh intervals, thresholds, etc.

5. **Run the Trading Bot**

   Run the main application to start fetching data, training the model, and executing trades based on predictions.

   ```bash
   python main.py
   ```

## Components

### 1. **Data Fetching**

- **Historical Data**: Fetches 2 years of historical data for specified assets from Yahoo Finance and saves it in the `data/` directory.
- **Real-Time Data**: Periodically fetches new data from MT5 to keep predictions accurate.

### 2. **Sentiment Analysis**

Uses the Bing API to fetch relevant news for each asset and applies sentiment analysis to influence trade decisions.

### 3. **Model Training**

The bot trains an LSTM model using historical price data and sentiment scores to make price predictions for each asset.

### 4. **Trade Execution**

Trade signals based on model predictions are sent to MT5, executing trades if the signal strength meets a specified threshold.

## Configuration

Modify the `config.yaml` file to adjust the following settings:

- **api_keys**: Keys for Bing API, MetaAPI, and MT5.
- **assets**: List of assets to be monitored and traded.
- **news_config**: Settings for news fetching interval.
- **model_parameters**: Configuration for model training intervals and prediction thresholds.

## Directory Structure

```plaintext
├── bot/
│   ├── data/                    # Contains historical and real-time data files
│   ├── models/                  # Directory for trained models
│   ├── config/config.yaml       # Configuration file for API keys and parameters
│   ├── logs/                    # Directory for log files
│   ├── preprocess_data.py       # Script to preprocess data for model training
│   ├── lstm_model.py            # Defines the LSTM model and training function
│   ├── main.py                  # Main script to start the bot
│   ├── news_scraper.py          # Fetches news data and performs sentiment analysis
│   ├── metaapi_trader.py        # Manages MT5 account connections and trade execution
└── requirements.txt             # List of Python dependencies
```

## Usage

To run individual components:

- **Historical Data Fetching**: `python fetch_historical_data.py`
- **News Scraping**: `python news_scraper.py`
- **Model Training**: `python lstm_model.py`
- **Main Trading Application**: `python main.py`

## Troubleshooting

- **API Errors**: Ensure API keys in `config.yaml` are correct and active.
- **Model Warnings**: Explicitly specify a model in `news_scraper.py` to avoid default model warnings.
- **Asyncio Errors**: Update Python packages to the latest versions to avoid deprecation warnings.

## License

MIT License. See `LICENSE` file for more information.

---
