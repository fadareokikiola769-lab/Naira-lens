from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import random
from datetime import datetime

app = Flask(__name__)
app.secret_key = "naira-lens-demo-key"

# HOME PAGE NOW REDIRECTS TO SIGNUP
@app.route("/")
def home():
    return redirect(url_for("signup"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        flash(f"Demo: Account created for {email}!", "success")
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        flash("Demo: Logged in successfully!", "success")
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    demo_data = {"naira_rate": 1580.50, "last_prediction": "BUY", "confidence": 87.3}
    return render_template("dashboard.html", data=demo_data)

@app.route("/predict", methods=["POST"])
def predict():
    prediction = {
        "status": "success",
        "prediction": random.choice(["BUY", "SELL", "HOLD"]),
        "confidence": round(random.uniform(65, 95), 2),
        "naira_rate": round(random.uniform(1500, 1650), 2),
        "sentiment": random.choice(["Positive", "Neutral", "Negative"]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
