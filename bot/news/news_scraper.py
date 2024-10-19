# news/news_scraper.py

import requests
import yaml
import os

class NewsScraper:
    def __init__(self, config_path="config/config.yaml"):
        # Load API key and assets list from config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.api_key = config['bing_api_key']
        self.assets = config['assets']
        self.api_url = "https://api.bing.microsoft.com/v7.0/news/search"

    def fetch_news(self, query, count=10):
        """Fetch news articles from Bing News API."""
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": count,
            "sortBy": "Date",
            "freshness": "Day",
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
        for item in news_data.get('value', []):
            articles.append({
                'name': item['name'],
                'url': item['url'],
                'description': item.get('description', ''),
                'datePublished': item['datePublished']
            })
        return articles

    def fetch_asset_news(self):
        """Fetch news for all assets defined in config."""
        all_news = {}
        for asset in self.assets:
            print(f"Fetching news for {asset}...")
            asset_news = self.fetch_news(query=asset)
            all_news[asset] = asset_news
        return all_news

# Example usage:
if __name__ == "__main__":
    scraper = NewsScraper()
    news = scraper.fetch_asset_news()
    for asset, articles in news.items():
        print(f"News for {asset}:")
        for article in articles:
            print(f"- {article['name']}: {article['url']}")

