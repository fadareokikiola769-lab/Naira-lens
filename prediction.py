"""
Feature Engineering & Fusion Layer + AI Prediction Model (3.5.4) +
Evaluation (3.6)

Builds three models per coin so the dashboard can show the comparison
the chapter's evaluation section calls for:
  - LSTM + sentiment  (the primary model, architecture per Figure 3.3)
  - GRU  + sentiment  (comparison architecture, Section 3.4.3)
  - LSTM, price only  (baseline, to quantify the contribution of sentiment)

Models are cached to disk per coin and retrained on a schedule
(Config.MODEL_RETRAIN_HOURS) rather than on every request, since
training on each API call would make the dashboard feel sluggish.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from config import Config
from extensions import db
from models import PredictionResult
from services.market_data import get_market_data
from services.sentiment import get_sentiment

# TensorFlow import is deferred into functions that need it so the rest of
# the app (auth, market data, sentiment) still works even before TF/Keras
# is installed, e.g. while you're wiring things up.


def _tf():
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
    return tf, layers, models, callbacks, optimizers


def _build_model(kind: str, n_features: int, window: int):
    _, layers, models, _, optimizers = _tf()
    Recurrent = layers.LSTM if kind == "lstm" else layers.GRU

    model = models.Sequential([
        layers.Input(shape=(window, n_features)),
        Recurrent(128, return_sequences=True),
        Recurrent(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=optimizers.Adam(), loss="mse")
    return model


def _make_windows(features: np.ndarray, target: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features[i:i + window])
        y.append(target[i + window])
    return np.array(X), np.array(y)


def _fuse_features(coin: str, force_refresh: bool = False):
    """
    Merges cleaned price data with the daily sentiment aggregate on the
    nearest date, per the join strategy described in Section 3.3.3.
    Returns a DataFrame indexed by date with columns [close, sentiment].
    """
    market_df = get_market_data(coin, force=force_refresh)
    _, daily_sentiment, vader_mean, roberta_mean, composite = get_sentiment(coin, force=force_refresh)

    price_daily = (
        market_df.assign(date=lambda d: pd.to_datetime(d["timestamp"]).dt.date)
        .groupby("date", as_index=False)["close"].last()
    )

    merged = price_daily.merge(daily_sentiment, on="date", how="left")
    # Nearest-timestamp join: forward/backward fill sentiment for days with
    # no matching news, then default any remaining gap to a neutral score.
    merged["score"] = merged["score"].ffill().bfill().fillna(0.0)
    merged = merged.rename(columns={"score": "sentiment"})
    merged = merged.sort_values("date").reset_index(drop=True)

    meta = {"vader_mean": vader_mean, "roberta_mean": roberta_mean, "composite": composite}
    return merged, meta


def _model_paths(coin: str, kind: str):
    stem = Config.MODELS_DIR / f"{coin}_{kind}"
    return {
        "model": stem.with_suffix(".keras"),
        "scaler_x": Path(f"{stem}_scaler_x.pkl"),
        "scaler_y": Path(f"{stem}_scaler_y.pkl"),
        "meta": Path(f"{stem}_meta.json"),
    }


def _is_model_fresh(meta_path: Path) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        trained_at = datetime.fromisoformat(meta["trained_at"])
    except Exception:
        return False
    return trained_at >= datetime.utcnow() - timedelta(hours=Config.MODEL_RETRAIN_HOURS)


def _train_one(coin: str, kind: str, df: pd.DataFrame, use_sentiment: bool, force: bool = False):
    """
    Trains (or loads a cached) model. `kind` is 'lstm' or 'gru'.
    Returns dict with model, scalers, test metrics, and test predictions.
    """
    _, _, models_mod, callbacks, _ = _tf()
    window = Config.WINDOW_SIZE
    tag = f"{kind}_{'sent' if use_sentiment else 'base'}"
    paths = _model_paths(coin, tag)

    feature_cols = ["close", "sentiment"] if use_sentiment else ["close"]
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    raw_x = df[feature_cols].to_numpy(dtype=float)
    raw_y = df[["close"]].to_numpy(dtype=float)
    scaled_x = scaler_x.fit_transform(raw_x)
    scaled_y = scaler_y.fit_transform(raw_y).flatten()

    X, y = _make_windows(scaled_x, scaled_y, window)
    if len(X) < 15:
        raise RuntimeError(
            f"Not enough history to train ({len(X)} samples) — increase MARKET_HISTORY_DAYS."
        )

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    use_cache = (not force) and _is_model_fresh(paths["meta"]) and paths["model"].exists()
    if use_cache:
        model = models_mod.load_model(paths["model"])
        scaler_x = joblib.load(paths["scaler_x"])
        scaler_y = joblib.load(paths["scaler_y"])
    else:
        model = _build_model(kind, n_features=len(feature_cols), window=window)
        es = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=60,
            batch_size=8,
            verbose=0,
            callbacks=[es],
        )
        paths["model"].parent.mkdir(parents=True, exist_ok=True)
        model.save(paths["model"])
        joblib.dump(scaler_x, paths["scaler_x"])
        joblib.dump(scaler_y, paths["scaler_y"])
        paths["meta"].write_text(json.dumps({"trained_at": datetime.utcnow().isoformat()}))

    # Evaluate on held-out test set
    if len(X_test) == 0:
        X_test, y_test = X_train[-3:], y_train[-3:]

    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    mape = float(mean_absolute_percentage_error(actual, pred) * 100)

    return {
        "model": model,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "feature_cols": feature_cols,
        "window": window,
        "metrics": {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 2)},
        "test_pred": pred.tolist(),
        "test_actual": actual.tolist(),
    }


def _forecast(bundle, last_rows: pd.DataFrame, horizon: int, sentiment_hint: float):
    """Recursively forecasts `horizon` future steps from the trained model."""
    model = bundle["model"]
    scaler_x, scaler_y = bundle["scaler_x"], bundle["scaler_y"]
    feature_cols = bundle["feature_cols"]
    window = bundle["window"]

    window_df = last_rows[feature_cols].tail(window).copy()
    scaled_window = scaler_x.transform(window_df.to_numpy(dtype=float))

    forecasts = []
    current = scaled_window.copy()
    for _ in range(horizon):
        x = current.reshape(1, window, len(feature_cols))
        next_scaled = float(model.predict(x, verbose=0).flatten()[0])
        next_price = float(scaler_y.inverse_transform([[next_scaled]])[0][0])
        forecasts.append(next_price)

        if len(feature_cols) == 2:
            next_row_raw = np.array([[next_price, sentiment_hint]])
        else:
            next_row_raw = np.array([[next_price]])
        next_row_scaled = scaler_x.transform(next_row_raw)
        current = np.vstack([current[1:], next_row_scaled])

    return forecasts


def run_pipeline(coin: str, force_refresh: bool = False, force_retrain: bool = False):
    """
    Orchestrates fusion -> train/load three models -> evaluate -> forecast.
    Returns a payload matching the dashboard's expected JSON contract.
    """
    df, sentiment_meta = _fuse_features(coin, force_refresh=force_refresh)
    horizon = Config.FORECAST_HORIZON_DAYS
    last_sentiment = float(df["sentiment"].tail(5).mean()) if len(df) else 0.0

    lstm_sent = _train_one(coin, "lstm", df, use_sentiment=True, force=force_retrain)
    gru_sent = _train_one(coin, "gru", df, use_sentiment=True, force=force_retrain)
    lstm_base = _train_one(coin, "lstm", df, use_sentiment=False, force=force_retrain)

    fc_lstm_sent = _forecast(lstm_sent, df, horizon, last_sentiment)
    fc_gru_sent = _forecast(gru_sent, df, horizon, last_sentiment)
    fc_lstm_base = _forecast(lstm_base, df, horizon, last_sentiment)

    # Persist a few forecast + recent test rows to prediction_results so the
    # "recent forecast log" table reflects real stored records (3.3.3).
    now = datetime.utcnow()
    log_rows = []
    for label, bundle in (("LSTM+Sentiment", lstm_sent), ("GRU+Sentiment", gru_sent), ("LSTM Baseline", lstm_base)):
        preds, actuals = bundle["test_pred"][-3:], bundle["test_actual"][-3:]
        for i, (p, a) in enumerate(zip(preds, actuals)):
            mape_i = abs((a - p) / a) * 100 if a else None
            row = PredictionResult(
                coin=coin,
                model_version=label,
                prediction_timestamp=now - timedelta(hours=len(preds) - i),
                predicted_value=p,
                actual_value=a,
                error_metric=mape_i,
            )
            db.session.add(row)
    db.session.commit()

    recent_logs = (
        PredictionResult.query.filter_by(coin=coin)
        .order_by(PredictionResult.prediction_timestamp.desc())
        .limit(9)
        .all()
    )

    horizon_labels = [f"+{i+1}d" for i in range(horizon)]
    history_dates = [d.isoformat() for d in df["date"]]

    return {
        "coin": coin,
        "market": {
            "dates": history_dates,
            "prices": df["close"].round(2).tolist(),
        },
        "sentiment": {
            "dates": history_dates,
            "scores": df["sentiment"].round(3).tolist(),
            "vader_mean": round(sentiment_meta["vader_mean"], 3),
            "roberta_mean": round(sentiment_meta["roberta_mean"], 3),
            "composite": round(sentiment_meta["composite"], 3),
        },
        "forecast": {
            "horizon_labels": horizon_labels,
            "lstm_sentiment": [round(v, 2) for v in fc_lstm_sent],
            "gru_sentiment": [round(v, 2) for v in fc_gru_sent],
            "baseline": [round(v, 2) for v in fc_lstm_base],
        },
        "evaluation": {
            "lstm_sentiment": lstm_sent["metrics"],
            "gru_sentiment": gru_sent["metrics"],
            "baseline": lstm_base["metrics"],
        },
        "log": [row.to_dict() for row in recent_logs],
    }
