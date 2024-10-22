import os
import yaml
import requests
import json
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NewsScraper:
    def __init__(self, config_path="config/config.yaml"):
        # Load API key and assets list from config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.api_key = config['bing_api_key']
        self.assets = config['assets']
        self.api_url = "https://api.bing.microsoft.com/v7.0/search"  # Bing Web Search API URL
        self.sentiment_file = 'sentiment.json'
        self.analyzer = SentimentIntensityAnalyzer()

    def fetch_news(self, query, count=10):
        """Fetch news articles from Bing Web Search API."""
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": count,
            "textFormat": "Raw",
            "mkt": "en-US"
        }
        
        response = requests.get(self.api_url, headers=headers, params=params)
        
        if response.status_code == 200:
            news_data = response.json()
            return self._parse_news(news_data)
        else:
            raise Exception(f"Failed to fetch news: {response.status_code}, {response.text}")
    
    def _parse_news(self, news_data):
        """Parse and return relevant news articles."""
        articles = []
        for item in news_data.get('webPages', {}).get('value', []):  # The 'webPages' field will have web search results
            articles.append({
                'name': item['name'],
                'url': item['url'],
                'description': item.get('snippet', ''),
                'datePublished': item.get('dateLastCrawled', 'Unknown')
            })
        return articles

    def analyze_sentiment_vader(self, news_text):
        """Analyzes the sentiment of the given text using VADER sentiment analysis."""
        sentiment_scores = self.analyzer.polarity_scores(news_text)
        sentiment = sentiment_scores['compound']  # Compound score is a general sentiment score
        confidence = max(sentiment_scores['pos'], sentiment_scores['neg'], sentiment_scores['neu'])

        # Define sentiment labels based on compound score
        if sentiment >= 0.05:
            return 1, confidence  # Positive sentiment
        elif sentiment <= -0.05:
            return -1, confidence  # Negative sentiment
        else:
            return 0, confidence  # Neutral sentiment

    def fetch_asset_news(self):
        """Fetch news for all assets defined in config and perform sentiment analysis."""
        all_news = {}
        for asset in self.assets:
            print(f"Fetching news for {asset}...")
            asset_news = self.fetch_news(query=asset)
            all_news[asset] = asset_news
        
        # Perform sentiment analysis and save to file
        self.save_sentiment_analysis(all_news)
    
    def save_sentiment_analysis(self, all_news):
        """Perform sentiment analysis on fetched news and save results to a file."""
        sentiment_data = []

        for asset, articles in all_news.items():
            for article in articles:
                sentiment, confidence = self.analyze_sentiment_vader(article['description'])
                sentiment_data.append({
                    'asset': asset,
                    'name': article['name'],
                    'url': article['url'],
                    'description': article['description'],
                    'datePublished': article['datePublished'],
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'timestamp': datetime.utcnow().isoformat()
                })

        # Remove old data (older than 7 days)
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        sentiment_data = [entry for entry in sentiment_data if datetime.fromisoformat(entry['timestamp']) > cutoff_date]

        # Sort by timestamp (most recent first)
        sentiment_data.sort(key=lambda x: x['timestamp'], reverse=True)

        # Save to file
        with open(self.sentiment_file, 'w') as f:
            json.dump(sentiment_data, f, indent=4)

# Example usage:
if __name__ == "__main__":
    scraper = NewsScraper()
    scraper.fetch_asset_news()
    print("Sentiment analysis completed and saved to sentiment.json.")

