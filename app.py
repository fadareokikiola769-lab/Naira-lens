from flask import Flask, request, jsonify, render_template
import random
from datetime import datetime
import os

app = Flask(__name__)

@app.route("/")
def home():
    # This will load your index.html from templates folder
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # DEMO MODE - returns fake data
    prediction = {
        "status": "success",
        "prediction": random.choice(["BUY", "SELL", "HOLD"]),
        "confidence": round(random.uniform(65, 95), 2),
        "naira_rate": round(random.uniform(1500, 1650), 2),
        "sentiment": random.choice(["Positive", "Neutral", "Negative"]),
        "message": "Demo data - Real ML coming soon",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
