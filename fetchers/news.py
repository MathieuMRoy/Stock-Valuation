"""
News fetcher - Fetch investor relations press releases
"""
import requests
import xml.etree.ElementTree as ET


def fetch_ir_press_releases(search_name: str) -> list:
    """
    Fetch investor relations press releases from Google News RSS.
    
    Args:
        search_name: Company name to search for
    
    Returns:
        List of news items with title, link, and publication date
    """
    try:
        query = f"{search_name} press release investor relations"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        news_items = []
        for item in root.findall(".//item")[:3]:
            news_items.append({
                "title": (item.find("title").text or "").strip(),
                "link": (item.find("link").text or "").strip(),
                "pubDate": (item.find("pubDate").text or "")[:16]
            })
        return news_items
    except:
        return []
