from datetime import datetime
from extensions import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(190), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    watchlist = db.Column(db.String(120), default="BTC,ETH")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "watchlist": self.watchlist.split(",") if self.watchlist else [],
            "created_at": self.created_at.isoformat(),
        }


class MarketData(db.Model):
    """
    Historical + real-time OHLC-style price records.
    Mirrors the market_data table described in Section 3.3.3:
    timestamp, open, close, high, low, volume, per coin.
    """
    __tablename__ = "market_data"

    id = db.Column(db.Integer, primary_key=True)
    coin = db.Column(db.String(10), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=True)

    __table_args__ = (db.UniqueConstraint("coin", "timestamp", name="uq_market_coin_ts"),)


class SentimentData(db.Model):
    """
    Sentiment_data table: source post identifier, timestamp, raw text,
    and computed sentiment polarity score (Section 3.3.3).
    """
    __tablename__ = "sentiment_data"

    id = db.Column(db.Integer, primary_key=True)
    coin = db.Column(db.String(10), nullable=False, index=True)
    post_id = db.Column(db.String(255), nullable=False)
    source = db.Column(db.String(60), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    raw_text = db.Column(db.Text, nullable=False)
    vader_score = db.Column(db.Float, nullable=True)
    roberta_score = db.Column(db.Float, nullable=True)
    compound_score = db.Column(db.Float, nullable=False)

    __table_args__ = (db.UniqueConstraint("coin", "post_id", name="uq_sentiment_coin_post"),)


class PredictionResult(db.Model):
    """
    Prediction_results table: model version, prediction timestamp,
    predicted value, actual value, and error metric (Section 3.3.3).
    """
    __tablename__ = "prediction_results"

    id = db.Column(db.Integer, primary_key=True)
    coin = db.Column(db.String(10), nullable=False, index=True)
    model_version = db.Column(db.String(60), nullable=False)
    prediction_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    target_timestamp = db.Column(db.DateTime, nullable=True)
    predicted_value = db.Column(db.Float, nullable=False)
    actual_value = db.Column(db.Float, nullable=True)
    error_metric = db.Column(db.Float, nullable=True)  # MAPE %, when actual is known

    def to_dict(self):
        return {
            "timestamp": self.prediction_timestamp.isoformat(sep=" ", timespec="minutes"),
            "coin": self.coin,
            "model": self.model_version,
            "predicted": round(self.predicted_value, 4),
            "actual": round(self.actual_value, 4) if self.actual_value is not None else None,
            "mape": round(self.error_metric, 2) if self.error_metric is not None else None,
        }
