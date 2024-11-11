import os
import yaml
import json
import aiohttp
import asyncio
from datetime import datetime, timedelta, timezone
import logging
import pandas as pd
from transformers import pipeline
from typing import Dict, List, Optional
import backoff
import sys

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
        self.last_fetch_times: Dict[str, datetime] = {}
        
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Initialize sentiment analysis
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment pipeline: {e}")
            sys.exit(1)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration with error handling and validation"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Validate required configuration fields
            required_fields = ['api_keys', 'assets', 'news_config']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required configuration field: {field}")
            
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _define_influencers(self) -> Dict[str, List[str]]:
        """Define comprehensive influencer keywords for each asset"""
        return {
            "EURUSD": [
                "ECB", "Eurozone", "EUR/USD", "European Central Bank",
                "EU economy", "Euro Dollar", "Federal Reserve", "ECB policy",
                "European inflation", "EU interest rates"
            ],
            "GBPUSD": [
                "BOE", "UK Economy", "GBP/USD", "Bank of England",
                "British pound", "UK inflation", "UK interest rates",
                "British economy", "Sterling Dollar", "UK monetary policy"
            ],
            "USDJPY": [
                "BOJ", "US-Japan Relations", "USD/JPY", "Bank of Japan",
                "Japanese economy", "Yen Dollar", "Japan interest rates",
                "Japan inflation", "Japanese monetary policy", "Fed BOJ"
            ],
            "XAUUSD": [
                "Gold prices", "Gold demand", "XAU/USD", "Gold market",
                "Gold trading", "Precious metals", "Gold reserves",
                "Gold investment", "Gold sentiment", "Gold technical analysis"
            ]
        }

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=5
    )
    async def _fetch_asset_news(
        self,
        session: aiohttp.ClientSession,
        query: str,
        headers: Dict[str, str],
        last_timestamp: Optional[datetime] = None
    ) -> List[Dict]:
        """Fetch news with improved error handling and rate limiting"""
        params = {
            "q": query,
            "customconfig": self.custom_config_id,
            "mkt": "en-US",
            "count": 50,
            "offset": 0,
            "freshness": "Day"
        }
        
        if last_timestamp:
            # Add time filter if we have a last timestamp
            min_time = last_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["freshness"] = f"min-time:{min_time}"

        all_results = []
        try:
            async with session.get(
                self.api_url,
                headers=headers,
                params=params,
                timeout=30
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
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
                else:
                    logger.error(f"API request failed with status {response.status}: {await response.text()}")
                    
            return all_results
        except Exception as e:
            logger.error(f"Error in _fetch_asset_news: {e}")
            raise

    def analyze_sentiment(self, text: str) -> dict:
        """Enhanced sentiment analysis with error handling"""
        try:
            result = self.sentiment_pipeline(text)[0]
            confidence = result['score']
            
            if result['label'] == 'POSITIVE':
                return {
                    'positive': confidence,
                    'negative': 0.0,
                    'neutral': 1.0 - confidence,
                    'compound': confidence
                }
            else:
                return {
                    'positive': 0.0,
                    'negative': confidence,
                    'neutral': 1.0 - confidence,
                    'compound': -confidence
                }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0}

    async def fetch_news(self):
        """Fetch news with improved error handling and rate limiting"""
        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        all_news = {}

        async with aiohttp.ClientSession() as session:
            tasks = []
            for asset in self.assets:
                last_fetch = self.last_fetch_times.get(asset)
                for topic in self.influencers.get(asset, []):
                    tasks.append(self._fetch_asset_news(
                        session,
                        f"{asset} {topic}",
                        headers,
                        last_fetch
                    ))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for asset, result in zip(self.assets, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch news for {asset}: {result}")
                    continue
                if result:
                    all_news[asset] = result
                self.last_fetch_times[asset] = datetime.now(timezone.utc)

        await self.process_and_save_news(all_news)

    async def process_and_save_news(self, all_news: Dict[str, List[Dict]]):
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

    def _update_sentiment_file(self, new_data: List[Dict]):
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

async def main():
    """Main function with proper cleanup"""
    try:
        scraper = NewsScraperEnhanced()
        await scraper.fetch_news()
        logger.info("News fetching completed successfully")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        sys.exit(1)
    finally:
        logger.info("News scraper finished")

if __name__ == "__main__":
    try:
        if sys.platform.startswith('win'):
            # Set up a proper event loop policy for Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        try:
            # Close all remaining tasks
            pending = asyncio.all_tasks(loop)
            loop.run_until_complete(asyncio.gather(*pending))
            
            # Stop the loop
            loop.stop()
            
            # Close the loop
            loop.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")