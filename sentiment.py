"""
Text Preprocessing + Sentiment Analysis Engine (3.5.2 / 3.5.3)

Data source note: the Twitter/X API used in the chapter's design now
requires a paid tier, so this build collects free, publicly available
crypto news RSS feeds (CoinDesk, Cointelegraph, Decrypt) as the
"social media / news" text source referenced in Section 3.3.1's
architecture. The preprocessing and scoring logic (VADER baseline +
optional transformer-based RoBERTa) matches Section 3.5.3 exactly —
swap in a Tweepy collector here without touching the rest of the
pipeline if you have Twitter/X API access.
"""
import re
from datetime import datetime, timedelta

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import Config
from extensions import db
from models import SentimentData

_vader = SentimentIntensityAnalyzer()
_roberta_pipeline = None  # lazy-loaded, only if Config.USE_ROBERTA


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_roberta():
    global _roberta_pipeline
    if _roberta_pipeline is None and Config.USE_ROBERTA:
        from transformers import pipeline
        _roberta_pipeline = pipeline(
            "sentiment-analysis", model=Config.ROBERTA_MODEL_NAME, truncation=True
        )
    return _roberta_pipeline


def _roberta_score(text: str) -> float:
    """Returns a polarity in [-1, 1] from the RoBERTa classifier."""
    clf = _get_roberta()
    if clf is None:
        return None
    result = clf(text[:512])[0]
    label = result["label"].lower()
    score = float(result["score"])
    if "neg" in label:
        return -score
    if "pos" in label:
        return score
    return 0.0


COIN_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "BNB": ["bnb", "binance coin", "binancecoin"],
}


def _fetch_headlines(coin: str, limit: int = 25):
    keywords = COIN_KEYWORDS.get(coin, [coin.lower()])
    items = []
    for url in Config.NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        for entry in feed.entries:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            combined = f"{title}. {summary}"
            lower = combined.lower()
            if any(k in lower for k in keywords) or coin.lower() in lower:
                published = getattr(entry, "published_parsed", None)
                ts = datetime(*published[:6]) if published else datetime.utcnow()
                post_id = getattr(entry, "id", None) or getattr(entry, "link", title)
                items.append(
                    {
                        "post_id": str(post_id),
                        "source": feed.feed.get("title", url),
                        "timestamp": ts,
                        "text": combined.strip(),
                    }
                )
    # Fall back to the general feed (unfiltered) if a coin has thin coverage,
    # so the demo always has content to score.
    if len(items) < 5:
        for url in Config.NEWS_FEEDS:
            try:
                feed = feedparser.parse(url)
            except Exception:
                continue
            for entry in feed.entries[:8]:
                title = getattr(entry, "title", "") or ""
                summary = getattr(entry, "summary", "") or ""
                published = getattr(entry, "published_parsed", None)
                ts = datetime(*published[:6]) if published else datetime.utcnow()
                post_id = getattr(entry, "id", None) or getattr(entry, "link", title)
                items.append(
                    {
                        "post_id": str(post_id),
                        "source": feed.feed.get("title", url),
                        "timestamp": ts,
                        "text": f"{title}. {summary}".strip(),
                    }
                )
    items.sort(key=lambda x: x["timestamp"], reverse=True)
    return items[:limit]


def _score_and_save(coin: str, items):
    saved = []
    existing = {
        row.post_id
        for row in SentimentData.query.filter_by(coin=coin).with_entities(SentimentData.post_id)
    }
    for it in items:
        clean = _clean_text(it["text"])
        if not clean:
            continue
        vader = _vader.polarity_scores(clean)["compound"]
        roberta = _roberta_score(clean) if Config.USE_ROBERTA else None
        compound = roberta if roberta is not None else vader

        record = {
            "post_id": it["post_id"],
            "source": it["source"][:60],
            "timestamp": it["timestamp"],
            "raw_text": it["text"][:500],
            "vader_score": vader,
            "roberta_score": roberta,
            "compound_score": compound,
        }
        saved.append(record)

        if it["post_id"] not in existing:
            db.session.add(
                SentimentData(
                    coin=coin,
                    post_id=it["post_id"],
                    source=record["source"],
                    timestamp=it["timestamp"],
                    raw_text=record["raw_text"],
                    vader_score=vader,
                    roberta_score=roberta,
                    compound_score=compound,
                )
            )
    db.session.commit()
    return saved


def _is_fresh(coin: str) -> bool:
    latest = (
        SentimentData.query.filter_by(coin=coin)
        .order_by(SentimentData.timestamp.desc())
        .first()
    )
    if not latest:
        return False
    return latest.timestamp >= datetime.utcnow() - timedelta(minutes=Config.SENTIMENT_CACHE_MINUTES)


def get_sentiment(coin: str, force: bool = False):
    """
    Returns (feed_items, daily_aggregate_df, vader_mean, roberta_mean, composite).
    daily_aggregate_df has columns [date, score] — the mean sentiment per day,
    matching the aggregation approach in Section 3.5.3 (there computed hourly;
    here daily, to align with the daily price granularity used for training).
    """
    if force or not _is_fresh(coin):
        headlines = _fetch_headlines(coin)
        if headlines:
            _score_and_save(coin, headlines)

    rows = (
        SentimentData.query.filter_by(coin=coin)
        .order_by(SentimentData.timestamp.desc())
        .limit(60)
        .all()
    )
    if not rows:
        return [], pd.DataFrame(columns=["date", "score"]), 0.0, 0.0, 0.0

    df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "source": r.source,
                "text": r.raw_text,
                "vader": r.vader_score,
                "roberta": r.roberta_score,
                "compound": r.compound_score,
            }
            for r in rows
        ]
    )
    df["date"] = df["timestamp"].dt.date
    daily = df.groupby("date", as_index=False)["compound"].mean().rename(columns={"compound": "score"})

    vader_mean = float(df["vader"].mean())
    roberta_vals = df["roberta"].dropna()
    roberta_mean = float(roberta_vals.mean()) if not roberta_vals.empty else vader_mean
    composite = float(df["compound"].mean())

    feed_items = [
        {
            "source": r["source"],
            "text": r["text"][:180],
            "timestamp": r["timestamp"].isoformat(),
            "score": round(r["compound"], 3),
            "label": "positive" if r["compound"] > 0.1 else "negative" if r["compound"] < -0.1 else "neutral",
        }
        for _, r in df.sort_values("timestamp", ascending=False).head(10).iterrows()
    ]

    return feed_items, daily, vader_mean, roberta_mean, composite
