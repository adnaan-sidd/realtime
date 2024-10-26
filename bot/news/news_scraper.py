import os
import yaml
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import logging
import pandas as pd
import pytz
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsScraper:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.subscription_key = self.config['api_keys']['bing_api_key']
        self.custom_config_id = self.config['api_keys']['bing_custom_config_id']
        self.assets = self.config['assets']
        self.api_url = "https://api.bing.microsoft.com/v7.0/custom/search"
        self.sentiment_file = 'data/sentiment.json'
        
        # Define the major influencers for each asset
        self.influencers = {
            "EURUSD": [
                "ECB monetary policy",
                "Fed monetary policy",
                "EU economic data",
                "US economic data",
                "political stability",
                "trade balances",
                "risk sentiment"
            ],
            "GBPUSD": [
                "BoE policy",
                "Fed policy",
                "Brexit-related events",
                "UK economic data",
                "US economic data",
                "political stability",
                "risk sentiment"
            ],
            "USDJPY": [
                "BoJ policy",
                "Fed policy",
                "interest rate differentials",
                "global risk sentiment",
                "trade data",
                "geopolitical events"
            ],
            "XAUUSD": [
                "inflation expectations",
                "USD strength",
                "interest rates",
                "central bank policies",
                "gold demand",
                "gold supply",
                "risk sentiment"
            ]
        }
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Initialize the sentiment analysis model
        self.sentiment_pipeline = pipeline("sentiment-analysis")

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
        except Exception as e:
            logger.error(f"Unexpected error loading config file: {e}")
            raise

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of given text using BERT model."""
        result = self.sentiment_pipeline(text)[0]
        if result['label'] == 'POSITIVE':
            return {'positive': result['score'], 'negative': 0.0, 'neutral': 1.0 - result['score']}
        elif result['label'] == 'NEGATIVE':
            return {'positive': 0.0, 'negative': result['score'], 'neutral': 1.0 - result['score']}
        else:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

    def fetch_news(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch news for all assets."""
        headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key
        }
        all_news = {}
        
        for asset in self.assets:
            asset_news = []
            for influencer in self.influencers.get(asset, []):
                query = f"{asset} {influencer}"
                news = self._fetch_asset_news(query, headers)
                if news:
                    asset_news.extend(news)
            if asset_news:
                all_news[asset] = asset_news
        
        self.process_and_save_news(all_news)
        return all_news

    def _fetch_asset_news(self, query: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
        """Fetch news for a single query using Bing Custom Search."""
        params = {
            "q": query,
            "customconfig": self.custom_config_id,
            "mkt": "en-US"
        }
        
        try:
            response = requests.get(self.api_url, headers=headers, params=params)
            logger.info(f"Requesting news for query: {query}")
            
            if response.status_code == 200:
                data = response.json()
                webpages = data.get('webPages', {}).get('value', [])
                return [
                    {
                        'title': webpage['name'],
                        'description': webpage.get('snippet', ''),
                        'url': webpage['url'],
                        'published': webpage.get('dateLastCrawled', ''),
                        'source': webpage.get('displayUrl', '').split('/')[0],
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    for webpage in webpages
                ]
            else:
                logger.error(f"Error fetching news for query: {query}: HTTP {response.status_code}")
                logger.error(f"Response content: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error while fetching news for query: {query}: {e}")
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
                    'source': article['source'],
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
            
            combined_data = new_data + existing_data
            
            with open(self.sentiment_file, 'w') as f:
                json.dump(combined_data, f, indent=4)
            
            logger.info(f"Updated sentiment file with {len(new_data)} new entries")
            
            # Convert JSON to CSV and save
            self.convert_json_to_csv()

        except Exception as e:
            logger.error(f"Error updating sentiment file: {e}")

    def convert_json_to_csv(self):
        """Convert the JSON sentiment file to CSV and keep data for a week."""
        try:
            # Load the data
            with open(self.sentiment_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df['positive'] = df['sentiment_scores'].apply(lambda x: x['positive'])
            df['negative'] = df['sentiment_scores'].apply(lambda x: x['negative'])
            df['neutral'] = df['sentiment_scores'].apply(lambda x: x['neutral'])

            # Filter data for the last week
            utc_now = datetime.now(pytz.utc)
            recent_df = df[df['timestamp'] > utc_now - timedelta(days=7)]

            # Save recent data to CSV
            recent_df.to_csv('data/sentiment_data.csv', index=False)
            logger.info("Data saved to 'data/sentiment_data.csv' for further analysis and plotting.")
        except Exception as e:
            logger.error(f"Error converting JSON to CSV: {e}")

if __name__ == "__main__":
    scraper = NewsScraper()
    scraper.fetch_news()
