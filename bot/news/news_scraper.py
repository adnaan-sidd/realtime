import os
import yaml
import json
import requests
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('news_scraper.log')
    ]
)
logger = logging.getLogger(__name__)

class NewsScraperEnhanced:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.subscription_key = self.config['api_keys']['bing_api_key']
        self.custom_config_id = self.config['api_keys']['bing_custom_config_id']
        self.assets = self.config['assets']
        self.api_url = "https://api.bing.microsoft.com/v7.0/custom/search"
        self.sentiment_file = 'data/sentiment.json'
        self.influencers = self._define_influencers()
        self.last_fetch_times = {}
        
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration with error handling and validation"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            required_fields = ['api_keys', 'assets', 'news_config']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required configuration field: {field}")
            
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _define_influencers(self) -> dict:
        """Define comprehensive influencer keywords for each asset"""
        return {
            "EURUSD": ["ECB", "Eurozone", "EUR/USD", "European Central Bank", "EU economy", "Euro Dollar", "Federal Reserve"],
            "GBPUSD": ["BOE", "UK Economy", "GBP/USD", "Bank of England", "British pound", "UK inflation"],
            "USDJPY": ["BOJ", "US-Japan Relations", "USD/JPY", "Bank of Japan", "Japanese economy", "Yen Dollar"],
            "XAUUSD": ["Gold prices", "Gold demand", "XAU/USD", "Gold market", "Gold trading"],
        }

    def _fetch_asset_news(self, query: str) -> list:
        """Fetch news with improved error handling"""
        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        params = {
            "q": query,
            "customconfig": self.custom_config_id,
            "mkt": "en-US",
            "count": 50,
            "offset": 0,
            "freshness": "Day"
        }
        
        all_results = []
        
        try:
            response = requests.get(self.api_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'webPages' in data and 'value' in data['webPages']:
                results = [{
                    'title': item['name'],
                    'description': item.get('snippet', ''),
                    'url': item['url'],
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'source': item.get('publisher', [{'name': 'Unknown'}])[0]['name']
                } for item in data['webPages']['value']]
                all_results.extend(results)
                logger.info(f"Successfully fetched {len(results)} articles for query: {query}")
            else:
                logger.warning(f"No results found for query: {query}")
        except Exception as e:
            logger.error(f"Error in _fetch_asset_news: {e}")
        
        return all_results

    def analyze_sentiment(self, text: str) -> dict:
        """Enhanced sentiment analysis with error handling"""
        # Mocked sentiment analysis since the sentiment model is removed
        # Replace with actual sentiment analysis logic if needed
        return {
            'positive': 0.5,
            'negative': 0.5,
            'neutral': 0.0,
            'compound': 0.0
        }

    def fetch_news(self):
        """Fetch news with error handling"""
        all_news = {}
        for asset in self.assets:
            for topic in self.influencers.get(asset, []):
                news_query = f"{asset} {topic}"
                news_articles = self._fetch_asset_news(news_query)
                if news_articles:
                    all_news[asset] = news_articles
                self.last_fetch_times[asset] = datetime.now(timezone.utc)

        self.process_and_save_news(all_news)

    def process_and_save_news(self, all_news: dict):
        """Process and save news with error handling"""
        sentiment_data = []
        try:
            for asset, articles in all_news.items():
                for article in articles:
                    text = f"{article['title']} {article['description']}"
                    sentiment_scores = self.analyze_sentiment(text)
                    
                    sentiment_data.append({
                        'asset': asset,
                        'title': article['title'],
                        'url': article['url'],
                        'source': article.get('source', 'Unknown'),
                        'sentiment_scores': sentiment_scores,
                        'timestamp': article['timestamp']
                    })
            
            # Save directly to file without using aiofiles
            self._update_sentiment_file(sentiment_data)
        except Exception as e:
            logger.error(f"Error processing news data: {e}")

    def _update_sentiment_file(self, new_data: list):
        """Update sentiment file with error handling"""
        try:
            existing_data = []
            if os.path.exists(self.sentiment_file):
                with open(self.sentiment_file, 'r') as f:
                    existing_data = json.load(f)

            # Remove old entries
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config['news_config']['retention_days'])
            existing_data = [
                entry for entry in existing_data 
                if datetime.fromisoformat(entry['timestamp']) > cutoff_date
            ]

            # Combine and deduplicate data
            seen_urls = set()
            combined_data = []
            for entry in new_data + existing_data:
                if entry['url'] not in seen_urls:
                    combined_data.append(entry)
                    seen_urls.add(entry['url'])

            # Save to file
            with open(self.sentiment_file, 'w') as f:
                json.dump(combined_data, indent=4, fp=f)

            logger.info(f"Updated sentiment file with {len(new_data)} new entries")
            self.convert_json_to_csv()
        except Exception as e:
            logger.error(f"Error updating sentiment file: {e}")

    def convert_json_to_csv(self):
        """Convert JSON to CSV with error handling"""
        try:
            with open(self.sentiment_file, 'r') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            df.to_csv('data/sentiment_data.csv', index=False)
            logger.info("Data successfully saved to 'data/sentiment_data.csv'")
        except Exception as e:
            logger.error(f"Error converting JSON to CSV: {e}")

def main():
    """Main function to run the scraper"""
    try:
        scraper = NewsScraperEnhanced()
        scraper.fetch_news()
        logger.info("News fetching completed successfully")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")

if __name__ == "__main__":
    main()