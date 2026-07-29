# NairaLens — AI Cryptocurrency Market Analysis

A working implementation of the system described in Chapter 3: a Flask API
that collects live price data (CoinGecko) and crypto news, scores it with
VADER (and optionally RoBERTa), fuses it with price history, trains an
LSTM/GRU forecast per coin, and serves it all to a static front end with
sign-up/sign-in.

```
cryptoai-site/
├── index.html            sign-in page
├── signup.html           sign-up page
├── dashboard.html         forecast dashboard
├── assets/
│   ├── css/style.css
│   └── js/ (config.js, auth.js, dashboard.js)
└── backend/                Flask API
    ├── app.py               routes
    ├── config.py             settings
    ├── models.py             DB schema (users, market_data, sentiment_data, prediction_results)
    ├── services/
    │   ├── market_data.py     collection + preprocessing (3.5.1 / 3.5.2)
    │   ├── sentiment.py       news retrieval + VADER/RoBERTa (3.5.3)
    │   └── prediction.py      feature fusion + LSTM/GRU + evaluation (3.5.4 / 3.6)
    ├── pretrain.py            optional: warm the model cache before first use
    └── requirements.txt
```

## 1. What's real vs. simplified

- **Prices** — live from the free CoinGecko API (no key needed).
- **Sentiment** — real VADER scoring of live crypto news RSS feeds
  (CoinDesk, Cointelegraph, Decrypt). The chapter specifies Twitter/X, but
  that API is now paid-only; RSS news is a drop-in substitute for the same
  role in the architecture. Swap in a Tweepy collector in
  `services/sentiment.py` if you have API access.
- **RoBERTa** — implemented but off by default (`USE_ROBERTA=false`)
  because `transformers`/`torch` are a large install. Turn it on in `.env`
  once you've installed them.
- **Prediction model** — an actual TensorFlow/Keras LSTM and GRU are
  trained on real fused price+sentiment data (Figure 3.3 architecture: two
  recurrent layers, dropout, dense ReLU, output). Training runs the first
  time you request a coin and is cached afterward (`MODEL_RETRAIN_HOURS`).
- **Granularity** — daily candles (not hourly), so a demo laptop can train
  in well under a minute. Configurable via `.env`.
- **Database** — SQLite by default so it runs with zero setup; a one-line
  swap to MySQL is documented below (matches Section 3.4.2's stack).

## 2. Requirements

- Python 3.10–3.12
- ~2 GB free disk (TensorFlow + dependencies)
- Internet access (CoinGecko + news feeds)

## 3. Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # then edit if needed (see below)
```

Edit `.env` if you want to:
- point `CORS_ORIGINS` at whatever port you'll serve the frontend from
- switch `DATABASE_URL` to MySQL
- turn on `USE_ROBERTA`

**Optional but recommended** — pre-train the models so your first
dashboard visit isn't waiting on training:

```bash
python pretrain.py
```

This trains LSTM+sentiment, GRU+sentiment, and the LSTM baseline for
BTC, ETH, SOL, and BNB, and caches them under `backend/data/models/`.
It takes a few minutes total; skip it and the dashboard will just train
on-demand (up to ~60s the first time you view a coin).

**Start the API:**

```bash
python app.py
```

You should see Flask running on `http://127.0.0.1:5000`. Leave this
terminal open.

## 4. Frontend setup

The frontend is static HTML/CSS/JS — serve it with any simple HTTP server
so the browser treats it as a real origin (not `file://`, which some
browsers block cookies/fetch on).

From the project root (not `backend/`), in a **second terminal**:

```bash
python3 -m http.server 8000
```

Then open **http://127.0.0.1:8000** in your browser.

If you serve the frontend from a different port, add it to
`CORS_ORIGINS` in `backend/.env` and restart the Flask server — otherwise
the browser will block the API requests.

## 5. Using it

1. Open `http://127.0.0.1:8000` → click **Create an account** (or
   **Continue with demo account** to skip straight in).
2. You'll land on the dashboard. Switching coins (BTC/ETH/SOL/BNB) calls
   `/api/dashboard/<coin>`, which runs the full pipeline: fetch price data
   → clean it → fetch & score news → fuse features → train/load LSTM &
   GRU → forecast → evaluate → return everything the charts need.
3. The **pipeline strip** at the top mirrors Figure 3.1's five stages.
4. The **forecast log** table reads real rows written to the
   `prediction_results` table.

## 6. Switching to MySQL (optional, per Section 3.4.2)

```bash
pip install pymysql
```

In `backend/.env`:
```
DATABASE_URL=mysql+pymysql://<user>:<password>@localhost/nairalens
```

Create the database once (`CREATE DATABASE nairalens;` in MySQL), then
start `python app.py` as normal — the tables are created automatically on
first run.

## 7. Common issues

| Symptom | Fix |
|---|---|
| "Can't reach the backend" on sign-in | Flask isn't running, or `NAIRALENS_API_BASE` in `assets/js/config.js` doesn't match its port/host. |
| Sign-in works but dashboard requests fail with a CORS error | Add your frontend's exact origin to `CORS_ORIGINS` in `.env` and restart Flask. |
| Dashboard is very slow on first load | Expected — it's training 3 models. Run `python pretrain.py` ahead of time, or just wait once per coin (cached after). |
| `RuntimeError: Not enough history to train` | Raise `MARKET_HISTORY_DAYS` in `.env`, or CoinGecko may be rate-limiting — wait a minute and retry. |
| Sentiment feed looks generic / not coin-specific | Coin-specific coverage in the RSS feeds was thin that day; the collector falls back to general market news so the pipeline still has text to score. |

## 8. Security note

This is a learning/demo build. Before using it beyond a local
demonstration: rotate `SECRET_KEY`, put real HTTPS in front of it, add
rate limiting on `/api/auth/*`, and don't expose `debug=True` in
production (`app.run(debug=True)` in `app.py`).
