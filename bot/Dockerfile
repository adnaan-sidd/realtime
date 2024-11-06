# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run the data collection and processing scripts in sequence
CMD ["sh", "-c", "python news/news_scraper.py && python candles.py && python yfinance.py && python preprocess_data.py && python models/lstm_model.py"]

