import os
import yaml
import json
import aiohttp
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import pandas as pd
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
        self.influencers = self._define_influencers()
        os.makedirs('data', exist_ok=True)
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _define_influencers(self):
        """Define influencer keywords for each asset."""
        return {
            "EURUSD": ["ECB", "Eurozone", "EUR/USD"],
            "GBPUSD": ["BOE", "UK Economy", "GBP/USD"],
            "USDJPY": ["BOJ", "US-Japan Relations", "USD/JPY"],
            "XAUUSD": ["Gold prices", "Gold demand", "XAU/USD"]
        }

    def analyze_sentiment(self, text: str) -> dict:
        result = self.sentiment_pipeline(text)[0]
        return {
            'positive': result['score'] if result['label'] == 'POSITIVE' else 0.0,
            'negative': result['score'] if result['label'] == 'NEGATIVE' else 0.0,
            'neutral': 1.0 - result['score']
        }

    async def fetch_news(self):
        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        all_news = {}
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_asset_news(session, f"{asset} {topic}", headers)
                     for asset in self.assets for topic in self.influencers.get(asset, [])]
            results = await asyncio.gather(*tasks)
            for asset, news in zip(self.assets, results):
                if news:
                    all_news[asset] = news
        self.process_and_save_news(all_news)

    async def _fetch_asset_news(self, session, query, headers, retries=3):
        params = {"q": query, "customconfig": self.custom_config_id, "mkt": "en-US"}
        for attempt in range(retries):
            try:
                async with session.get(self.api_url, headers=headers, params=params) as response:
                    logger.info(f"Request URL: {response.url}")
                    logger.info(f"Response Status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        return [{
                            'title': item['name'],
                            'description': item.get('snippet', ''),
                            'url': item['url'],
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        } for item in data.get('webPages', {}).get('value', [])]
                    else:
                        logger.error(f"Failed request for {query}, attempt {attempt + 1}, status: {response.status}")
            except Exception as e:
                logger.error(f"Error fetching news for {query}: {e}")
            await asyncio.sleep(2)
        return []

    def process_and_save_news(self, all_news):
        sentiment_data = []
        for asset, articles in all_news.items():
            for article in articles:
                sentiment_scores = self.analyze_sentiment(article['title'] + " " + article['description'])
                sentiment_data.append({
                    'asset': asset,
                    'title': article['title'],
                    'url': article['url'],
                    'sentiment_scores': sentiment_scores,
                    'timestamp': article['timestamp']
                })
        self._update_sentiment_file(sentiment_data)

    def _update_sentiment_file(self, new_data):
        try:
            if os.path.exists(self.sentiment_file):
                with open(self.sentiment_file, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.config['news_config']['retention_days'])
            existing_data = [entry for entry in existing_data if datetime.fromisoformat(entry['timestamp']) > cutoff_date]
            combined_data = new_data + existing_data

            with open(self.sentiment_file, 'w') as f:
                json.dump(combined_data, f, indent=4)
            logger.info(f"Updated sentiment file with {len(new_data)} new entries")
            self.convert_json_to_csv()
        except Exception as e:
            logger.error(f"Error updating sentiment file: {e}")

    def convert_json_to_csv(self):
        try:
            with open(self.sentiment_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df.to_csv('data/sentiment_data.csv', index=False)
            logger.info("Data saved to 'data/sentiment_data.csv'")
        except Exception as e:
            logger.error(f"Error converting JSON to CSV: {e}")

if __name__ == "__main__":
    scraper = NewsScraper()
    asyncio.run(scraper.fetch_news())

