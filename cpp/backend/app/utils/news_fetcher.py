import requests
import os
from datetime import datetime, timedelta

# You can set your NewsAPI key here or use an environment variable
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '8b25fe4c845b4ed2a25caea16f8d7abd')
NEWSAPI_URL = 'https://newsapi.org/v2/everything'

# Default keywords for crops/mandis in Karnataka
DEFAULT_KEYWORDS = [
    'Arecanut Karnataka',
    'Coconut Karnataka',
    'Pepper Karnataka',
    'Cardamom Karnataka',
    'Elaichi Karnataka',
    'Mandi price Karnataka',
    'Crop price Karnataka',
]


def fetch_news(keywords=None, from_days_ago=7, language='en', max_articles=20):
    """
    Fetch news articles from NewsAPI for the given keywords from the last N days.
    Returns a list of articles (dicts).
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    all_articles = []
    from_date = (datetime.now() - timedelta(days=from_days_ago)).strftime('%Y-%m-%d')
    for keyword in keywords:
        params = {
            'q': keyword,
            'from': from_date,
            'language': language,
            'sortBy': 'publishedAt',
            'apiKey': NEWSAPI_KEY,
            'pageSize': max_articles,
        }
        resp = requests.get(NEWSAPI_URL, params=params)
        if resp.status_code == 200:
            data = resp.json()
            if 'articles' in data:
                for article in data['articles']:
                    article['search_keyword'] = keyword
                all_articles.extend(data['articles'])
        else:
            print(f"NewsAPI error for '{keyword}': {resp.status_code} {resp.text}")
    # Remove duplicates by url
    seen = set()
    unique_articles = []
    for art in all_articles:
        if art['url'] not in seen:
            unique_articles.append(art)
            seen.add(art['url'])
    return unique_articles


def extract_news_features(articles, keywords=None):
    """
    Extract simple features from news articles for ML:
    - article_count: total number of articles
    - keyword_counts: dict of keyword -> count
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    keyword_counts = {k: 0 for k in keywords}
    for art in articles:
        k = art.get('search_keyword')
        if k in keyword_counts:
            keyword_counts[k] += 1
    features = {
        'news_article_count': len(articles),
        'news_keyword_counts': keyword_counts,
    }
    return features


if __name__ == "__main__":
    # Test fetch
    news = fetch_news()
    print(f"Fetched {len(news)} articles.")
    feats = extract_news_features(news)
    print(feats) 