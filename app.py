import random
from datetime import datetime

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # This is DEMO MODE - No pandas needed
    # It just returns realistic fake data
    
    prediction = {
        "status": "success",
        "prediction": random.choice(["BUY", "SELL", "HOLD"]),
        "confidence": round(random.uniform(65, 95), 2),
        "naira_rate": round(random.uniform(1500, 1650), 2),
        "sentiment": random.choice(["Positive", "Neutral", "Negative"]),
        "message": "This is demo data. Connect real ML model later.",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return prediction
