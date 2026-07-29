"""
NairaLens backend — Flask API implementing the system in Chapter 3:
data collection, preprocessing, sentiment analysis, LSTM/GRU prediction,
and the endpoints the dashboard renders (evaluation, visualisation).

Run with:  python app.py     (see README.md for full setup)
"""
from datetime import datetime

from flask import Flask, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config
from extensions import db, cors
from models import User


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    cors.init_app(
        app,
        supports_credentials=True,
        origins=Config.CORS_ORIGINS,
    )

    with app.app_context():
        db.create_all()

    register_routes(app)
    return app


def _current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return User.query.get(uid)


def register_routes(app):

    # ---------------------------------------------------------------- health
    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

    # ------------------------------------------------------------------ auth
    @app.post("/api/auth/signup")
    def signup():
        data = request.get_json(silent=True) or {}
        name = (data.get("name") or "").strip()
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""
        watchlist = data.get("watchlist") or ["BTC", "ETH"]

        if len(name) < 2:
            return jsonify({"error": "Enter your full name."}), 400
        if "@" not in email or "." not in email:
            return jsonify({"error": "Enter a valid email address."}), 400
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters."}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "An account with this email already exists."}), 409

        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            watchlist=",".join(watchlist),
        )
        db.session.add(user)
        db.session.commit()

        session["user_id"] = user.id
        return jsonify({"user": user.to_dict()}), 201

    @app.post("/api/auth/login")
    def login():
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Incorrect email or password."}), 401

        session["user_id"] = user.id
        return jsonify({"user": user.to_dict()})

    @app.post("/api/auth/logout")
    def logout():
        session.clear()
        return jsonify({"ok": True})

    @app.get("/api/auth/me")
    def me():
        user = _current_user()
        if not user:
            return jsonify({"error": "Not authenticated."}), 401
        return jsonify({"user": user.to_dict()})

    # --------------------------------------------------------------- pipeline
    def _require_login():
        user = _current_user()
        if not user:
            return None, (jsonify({"error": "Not authenticated."}), 401)
        return user, None

    @app.get("/api/market/<coin>")
    def market(coin):
        from services.market_data import get_market_data
        _, err = _require_login()
        if err:
            return err
        try:
            df = get_market_data(coin.upper())
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({
            "coin": coin.upper(),
            "dates": df["timestamp"].astype(str).tolist(),
            "prices": df["close"].round(2).tolist(),
        })

    @app.get("/api/sentiment/<coin>")
    def sentiment(coin):
        from services.sentiment import get_sentiment
        _, err = _require_login()
        if err:
            return err
        try:
            feed, daily, vader_mean, roberta_mean, composite = get_sentiment(coin.upper())
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({
            "coin": coin.upper(),
            "feed": feed,
            "daily": [{"date": str(d), "score": s} for d, s in zip(daily.get("date", []), daily.get("score", []))],
            "vader_mean": round(vader_mean, 3),
            "roberta_mean": round(roberta_mean, 3),
            "composite": round(composite, 3),
        })

    @app.get("/api/dashboard/<coin>")
    def dashboard(coin):
        from services.prediction import run_pipeline
        user, err = _require_login()
        if err:
            return err
        coin = coin.upper()
        if coin not in Config.COINGECKO_IDS:
            return jsonify({"error": f"Unsupported coin: {coin}"}), 400

        force_refresh = request.args.get("refresh") == "1"
        force_retrain = request.args.get("retrain") == "1"
        try:
            payload = run_pipeline(coin, force_refresh=force_refresh, force_retrain=force_retrain)
        except Exception as e:
            return jsonify({"error": f"Pipeline failed: {e}"}), 500
        return jsonify(payload)

    @app.get("/api/log/<coin>")
    def log(coin):
        from models import PredictionResult
        _, err = _require_login()
        if err:
            return err
        rows = (
            PredictionResult.query.filter_by(coin=coin.upper())
            .order_by(PredictionResult.prediction_timestamp.desc())
            .limit(20)
            .all()
        )
        return jsonify({"log": [r.to_dict() for r in rows]})


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
