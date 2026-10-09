import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from src.predict import SpamPredictor

app = Flask(__name__)

# Configure DB
database_url = os.environ.get("DATABASE_URL")
if database_url:
    # SQLAlchemy 1.4+ requires postgresql:// instead of postgres://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///spam.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# Load predictor once at startup
predictor = SpamPredictor()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in JSON payload"}), 400
    
    text = data["text"]
    if not text or not text.strip():
        return jsonify({"error": "Text cannot be empty"}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text exceeds 5000 characters limit"}), 400
        
    try:
        res = predictor.predict_one(text)
        
        # Save to DB
        pred_record = Prediction(
            message=text,
            label=res["label"],
            confidence=res["confidence"]
        )
        db.session.add(pred_record)
        db.session.commit()
        
        return jsonify({
            "label": res["label"],
            "confidence": res["confidence"],
            "spam_probability": res["spam_probability"],
            "ham_probability": res["ham_probability"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history")
def history():
    try:
        # Show last 10 checks
        recent_checks = Prediction.query.order_by(Prediction.id.desc()).limit(10).all()
        history_data = []
        for check in recent_checks:
            history_data.append({
                "id": check.id,
                "message": check.message,
                "label": check.label,
                "confidence": check.confidence,
                "created_at": check.created_at.isoformat()
            })
        return jsonify(history_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
