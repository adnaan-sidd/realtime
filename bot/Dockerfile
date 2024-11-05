# Use Python 3.10 as the base image
FROM python:3.10

# Set a working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Ensure the logs directory exists for logging output
RUN mkdir -p /app/logs

# Run the news scraper and data fetching scripts sequentially
CMD ["sh", "-c", "python news/news_scraper.py && python fetch_historical_data.py"]

