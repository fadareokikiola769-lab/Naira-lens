from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html") # Your frontend will still show

@app.route("/predict", methods=["POST"])
def predict():
    return {"message": "ML is temporarily disabled. Full version coming in 1 hour", "status": "demo"}

if __name__ == "__main__":
    app.run()
