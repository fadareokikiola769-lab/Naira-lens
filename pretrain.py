"""
Optional: pre-train and cache the LSTM/GRU models for every supported coin
before starting the server, so the first dashboard request per coin is
instant instead of waiting ~30-60s for training.

Usage:
    python pretrain.py
"""
from app import create_app
from config import Config
from services.prediction import run_pipeline

if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        for coin in Config.COINGECKO_IDS:
            print(f"Training models for {coin} ...")
            try:
                run_pipeline(coin, force_refresh=True, force_retrain=True)
                print(f"  done: {coin}")
            except Exception as e:
                print(f"  failed for {coin}: {e}")
    print("Pretraining complete.")
