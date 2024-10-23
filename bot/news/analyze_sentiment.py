import json
import pandas as pd
from datetime import datetime, timedelta
import pytz

def analyze_sentiment_data(file_path='data/sentiment.json'):
    # Load the data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['positive'] = df['sentiment_scores'].apply(lambda x: x['positive'])
    df['negative'] = df['sentiment_scores'].apply(lambda x: x['negative'])
    df['neutral'] = df['sentiment_scores'].apply(lambda x: x['neutral'])

    # Basic statistics
    print(f"Total entries: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Assets: {', '.join(df['asset'].unique())}")

    # Recent sentiment analysis
    utc_now = datetime.now(pytz.utc)
    recent_df = df[df['timestamp'] > utc_now - timedelta(days=7)]
    
    for asset in recent_df['asset'].unique():
        asset_df = recent_df[recent_df['asset'] == asset]
        print(f"\nRecent Sentiment for {asset}:")
        print(f"  Positive: {asset_df['positive'].mean():.2f}")
        print(f"  Negative: {asset_df['negative'].mean():.2f}")
        print(f"  Neutral: {asset_df['neutral'].mean():.2f}")

    # Save data to CSV for external plotting
    df.to_csv('data/sentiment_data.csv', index=False)
    print("\nData saved to 'data/sentiment_data.csv' for further analysis and plotting.")

if __name__ == "__main__":
    analyze_sentiment_data()
