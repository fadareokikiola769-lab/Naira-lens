from flask import Flask, request, jsonify
import random
from datetime import datetime

app = Flask(__name__) # THIS LINE WAS MISSING

@app.route("/")
def home():
    return "Naira Lens API is Live! Go to /predict to test"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # DEMO MODE - No pandas needed
    prediction = {
        "status": "success",
        "prediction": random.choice(["BUY", "SELL", "HOLD"]),
        "confidence": round(random.uniform(65, 95), 2),
        "naira_rate": round(random.uniform(1500, 1650), 2),
        "sentiment": random.choice(["Positive", "Neutral", "Negative"]),
        "message": "This is demo data. Connect real ML model later.",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
