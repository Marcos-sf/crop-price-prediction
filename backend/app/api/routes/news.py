from fastapi import APIRouter, Query
from typing import List, Optional
from app.utils.news_fetcher import fetch_news

router = APIRouter()

@router.get("/news", tags=["News"])
def get_news(
    keywords: Optional[List[str]] = Query(None, description="Keywords to search for"),
    days: int = Query(7, ge=1, le=30, description="How many days back to fetch news")
):
    """
    Get recent news headlines for given keywords (default: crop/mandi keywords).
    """
    articles = fetch_news(keywords=keywords, from_days_ago=days)
    # Return only relevant fields for frontend
    result = [
        {
            "title": art.get("title"),
            "url": art.get("url"),
            "publishedAt": art.get("publishedAt"),
            "source": art.get("source", {}).get("name"),
            "description": art.get("description"),
            "search_keyword": art.get("search_keyword"),
        }
        for art in articles
    ]
    return {"articles": result} 