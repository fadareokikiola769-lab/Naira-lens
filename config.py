"""
NairaLens backend configuration.

All values can be overridden with environment variables (see .env.example).
Defaults are chosen so the system runs out of the box with SQLite and no
paid API keys, while remaining a straightforward swap to MySQL per the
database design described in Chapter 3 (Section 3.3.3).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _env_bool(name, default=False):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

    # --- Database -----------------------------------------------------
    # Default: SQLite file, zero setup.
    # MySQL (as specified in Chapter 3, Section 3.4.2):
    #   DATABASE_URL=mysql+pymysql://<user>:<password>@localhost/nairalens
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{(DATA_DIR / 'nairalens.db').as_posix()}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- CORS -----------------------------------------------------------
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500,"
        "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",")

    # --- Sessions ---------------------------------------------------------
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_HTTPONLY = True

    # --- Coins supported --------------------------------------------------
    COINGECKO_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binancecoin",
    }
    # >90 triggers CoinGecko's daily granularity automatically, which keeps
    # the training set small (~120 points) and training fast for a demo.
    MARKET_HISTORY_DAYS = int(os.getenv("MARKET_HISTORY_DAYS", "120"))
    FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "7"))
    WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "10"))  # sliding window (time steps)

    # --- Sentiment ----------------------------------------------------------
    NEWS_FEEDS = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
    ]
    USE_ROBERTA = _env_bool("USE_ROBERTA", False)
    ROBERTA_MODEL_NAME = os.getenv(
        "ROBERTA_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    # --- Caching / freshness -----------------------------------------------
    MARKET_CACHE_MINUTES = int(os.getenv("MARKET_CACHE_MINUTES", "30"))
    SENTIMENT_CACHE_MINUTES = int(os.getenv("SENTIMENT_CACHE_MINUTES", "30"))
    MODEL_RETRAIN_HOURS = int(os.getenv("MODEL_RETRAIN_HOURS", "24"))
