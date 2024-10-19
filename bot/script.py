import os

# Define the directory structure
structure = {
    "data": [
        "EURUSD_data.csv",
        "GBPUSD_data.csv",
        "JPYUSD_data.csv",
        "XAUUSD_data.csv"
    ],
    "config": [
        "config.yaml"
    ],
    "indicators": [
        "indicators.py"
    ],
    "strategies": [
        "moving_average.py",
        "rsi_strategy.py",
        "bollinger_bands.py"
    ],
    "models": [
        "lstm_model.py"
    ],
    "news": [
        "news_scraper.py"
    ],
    ".": [  # root directory files
        "metaapi_trader.py",
        "preprocess_data.py",
        "backtest.py",
        "portfolio.py",
        "main.py",
        "requirements.txt",
        "README.md"
    ]
}

# Create directories and files
for folder, files in structure.items():
    # If the folder is '.', it means the base directory
    if folder != ".":
        os.makedirs(folder, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(folder, file) if folder != "." else file
        # Create empty files
        with open(file_path, 'w') as f:
            pass

print("Directory structure created successfully.")
