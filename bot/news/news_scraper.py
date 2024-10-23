import os
import yaml
import json
import requests
from textblob import TextBlob
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsScraper:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.api_key = self.config['api_keys']['bing_api_key']
        self.assets = self.config['assets']
        self.api_url = "https://api.bing.microsoft.com/v7.0/search"
        self.sentiment_file = 'data/sentiment.json'
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of given text using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0:
            return {'positive': polarity, 'negative': 0, 'neutral': 1 - polarity}
        elif polarity < 0:
            return {'positive': 0, 'negative': -polarity, 'neutral': 1 + polarity}
        else:
            return {'positive': 0, 'negative': 0, 'neutral': 1}

    def fetch_news(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch news for all assets."""
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        all_news = {}
        
        for asset in self.assets:
            news = self._fetch_asset_news(asset, headers)
            if news:
                all_news[asset] = news
        
        self.process_and_save_news(all_news)
        return all_news

    def _fetch_asset_news(self, asset: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
        """Fetch news for a single asset."""
        params = {
            "q": f"{asset} forex trading news",
            "count": 50,
            "freshness": "Day",
            "mkt": "en-US",
            "responseFilter": "Webpages"
        }
        
        try:
            response = requests.get(self.api_url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                webpages = data.get('webPages', {}).get('value', [])
                return [
                    {
                        'title': webpage['name'],
                        'description': webpage.get('snippet', ''),
                        'url': webpage['url'],
                        'published': webpage.get('dateLastCrawled', ''),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    for webpage in webpages
                ]
            else:
                logger.error(f"Error fetching news for {asset}: HTTP {response.status_code}")
                logger.error(f"Response content: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error while fetching news for {asset}: {e}")
            return []

    def process_and_save_news(self, all_news: Dict[str, List[Dict[str, Any]]]):
        """Process fetched news and save sentiment data."""
        sentiment_data = []
        
        for asset, articles in all_news.items():
            for article in articles:
                sentiment_scores = self.analyze_sentiment(
                    article['title'] + " " + article['description']
                )
                
                sentiment_data.append({
                    'asset': asset,
                    'title': article['title'],
                    'url': article['url'],
                    'sentiment_scores': sentiment_scores,
                    'timestamp': article['timestamp']
                })
        
        self._update_sentiment_file(sentiment_data)

    def _update_sentiment_file(self, new_data: List[Dict[str, Any]]):
        """Update sentiment file with new data and remove old entries."""
        try:
            if os.path.exists(self.sentiment_file):
                with open(self.sentiment_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config['news_config']['retention_days'])
            existing_data = [
                entry for entry in existing_data 
                if datetime.fromisoformat(entry['timestamp']) > cutoff_date
            ]
            
            existing_data.extend(new_data)
            
            with open(self.sentiment_file, 'w') as f:
                json.dump(existing_data, f, indent=4)
            
            logger.info(f"Updated sentiment file with {len(new_data)} new entries")
        except Exception as e:
            logger.error(f"Error updating sentiment file: {e}")

    def run_continuous(self):
        """Run the news scraper continuously."""
        while True:
            try:
                self.fetch_news()
                logger.info("Completed news fetch and analysis")
                time.sleep(3600)  # Sleep for 1 hour
            except Exception as e:
                logger.error(f"Error in run_continuous: {e}")
                time.sleep(60)  # Wait for 1 minute before retrying

if __name__ == "__main__":
    scraper = NewsScraper()
    scraper.run_continuous()
