from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_newsapi_news(
    query: Annotated[str, "Search query, e.g. 'AAPL Apple earnings'"],
    from_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    to_date: Annotated[str, "End date in yyyy-mm-dd format"],
    max_articles: Annotated[int, "Maximum number of articles to return"] = 10,
) -> str:
    """
    Fetch recent news articles from NewsAPI.org for a given query.
    Requires NEWSAPI_KEY environment variable.
    Returns article titles, descriptions, sources, and URLs.
    """
    import os, requests
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        return "NEWSAPI_KEY not configured. Add it to .env to enable NewsAPI."
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "from": from_date,
                "to": to_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": max_articles,
                "apiKey": api_key,
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("status") != "ok":
            return f"NewsAPI error: {data.get('message', 'unknown error')}"
        articles = data.get("articles", [])
        if not articles:
            return f"No NewsAPI articles found for '{query}' between {from_date} and {to_date}."
        lines = [f"=== NewsAPI: {len(articles)} articles for '{query}' ===\n"]
        for a in articles:
            lines.append(f"**{a['title']}**")
            lines.append(f"Source: {a.get('source', {}).get('name', '')} | Date: {a.get('publishedAt', '')[:10]}")
            if a.get("description"):
                lines.append(a["description"])
            lines.append(f"URL: {a['url']}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"NewsAPI fetch error: {e}"


@tool
def get_stocktwits_sentiment(
    ticker: Annotated[str, "Ticker symbol (e.g. AAPL)"],
) -> str:
    """
    Fetch recent StockTwits messages and community sentiment for a ticker.
    Returns bullish/bearish counts and the most recent posts.
    No API key required.
    """
    import requests
    try:
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json?limit=30"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = resp.json()
        messages = data.get("messages", [])
        if not messages:
            return f"No StockTwits data found for {ticker}."
        bullish = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment") and
               m["entities"]["sentiment"].get("basic") == "Bullish"
        )
        bearish = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment") and
               m["entities"]["sentiment"].get("basic") == "Bearish"
        )
        neutral = len(messages) - bullish - bearish
        lines = [
            f"=== StockTwits Sentiment for {ticker} (last {len(messages)} posts) ===",
            f"Bullish: {bullish} | Bearish: {bearish} | Neutral: {neutral}",
            f"Sentiment ratio: {bullish/(bullish+bearish)*100:.0f}% bullish" if (bullish+bearish) > 0 else "",
            "\nRecent posts:",
        ]
        for m in messages[:12]:
            sent = ((m.get("entities") or {}).get("sentiment") or {}).get("basic", "Neutral")
            lines.append(f"[{sent}] {m['body'][:200]}")
        return "\n".join(lines)
    except Exception as e:
        return f"StockTwits fetch error: {e}"


@tool
def get_reddit_sentiment(
    ticker: Annotated[str, "Ticker symbol (e.g. AAPL)"],
    lookback_days: Annotated[int, "Number of days to look back"] = 7,
) -> str:
    """
    Fetch recent Reddit posts about a ticker from r/stocks, r/wallstreetbets, r/investing.
    Returns posts sorted by score (upvotes) with titles and excerpts.
    No API key required.
    """
    import requests
    from datetime import datetime, timedelta
    try:
        subreddits = ["stocks", "wallstreetbets", "investing"]
        all_posts = []
        headers = {"User-Agent": "TradingAgents/1.0 (research bot)"}
        cutoff = datetime.now() - timedelta(days=lookback_days)

        for sub in subreddits:
            try:
                resp = requests.get(
                    f"https://www.reddit.com/r/{sub}/search.json",
                    params={"q": ticker, "sort": "new", "limit": 15, "t": "week", "restrict_sr": 1},
                    headers=headers,
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                for p in resp.json().get("data", {}).get("children", []):
                    d = p["data"]
                    created = datetime.fromtimestamp(d.get("created_utc", 0))
                    if created < cutoff:
                        continue
                    all_posts.append({
                        "sub": sub,
                        "title": d.get("title", ""),
                        "score": d.get("score", 0),
                        "comments": d.get("num_comments", 0),
                        "text": (d.get("selftext") or "")[:300],
                        "date": created.strftime("%Y-%m-%d"),
                    })
            except Exception:
                continue

        if not all_posts:
            return f"No Reddit posts found for {ticker} in the last {lookback_days} days."

        all_posts.sort(key=lambda x: x["score"], reverse=True)
        lines = [f"=== Reddit Sentiment for {ticker} (last {lookback_days}d | {len(all_posts)} posts) ===\n"]
        for p in all_posts[:15]:
            lines.append(
                f"[r/{p['sub']}] {p['title']}"
                f"  (↑{p['score']} | 💬{p['comments']} | {p['date']})"
            )
            if p["text"]:
                lines.append(f"  {p['text'][:200]}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Reddit fetch error: {e}"

@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return route_to_vendor("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor.
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 7)
        limit (int): Maximum number of articles to return (default 5)
    Returns:
        str: A formatted string containing global news data
    """
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker)
