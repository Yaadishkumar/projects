"""CLI and interactive prediction interface for Email/SMS Spam Classifier.

Usage:
  # Single prediction via argument:
  py src/predict.py --text "Congratulations! You have won a $1,000 Walmart gift card. Call 08001234 now!"

  # Interactive mode:
  py src/predict.py --interactive
"""

import os
import sys
import argparse
import joblib
import numpy as np

# Ensure src modules are resolvable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text, download_nltk_resources

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.joblib")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.joblib")


class SpamPredictor:
    """Predictor class encapsulating vectorizer, model, and inference logic."""

    def __init__(self, model_path: str = MODEL_PATH, vectorizer_path: str = VECTORIZER_PATH):
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Model or vectorizer not found!\n"
                f"Expected:\n  - {model_path}\n  - {vectorizer_path}\n"
                f"Please run 'src/train.py' first to train and save the model."
            )

        print("[Predictor] Loading model and vectorizer...")
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.model_name = type(self.model).__name__
        download_nltk_resources()

    def predict_one(self, raw_text: str) -> dict:
        """Classify a single raw text string as SPAM or HAM with confidence.

        Args:
            raw_text: Raw message text.

        Returns:
            Dict containing label, confidence, probability_spam, cleaned_text, and matched_features.
        """
        cleaned = clean_text(raw_text)

        if not cleaned.strip():
            # Handle empty or stripped input gracefully
            return {
                "label": "HAM",
                "is_spam": False,
                "confidence": 0.50,
                "spam_probability": 0.50,
                "cleaned_text": cleaned,
                "matched_features": [],
            }

        # Vectorize
        X_vec = self.vectorizer.transform([cleaned])

        # Compute probability / confidence
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_vec)[0]
            prob_ham, prob_spam = probs[0], probs[1]
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X_vec)[0]
            # Sigmoid Platt-style calibration for LinearSVC
            prob_spam = 1.0 / (1.0 + np.exp(-decision))
            prob_ham = 1.0 - prob_spam
        else:
            pred = self.model.predict(X_vec)[0]
            prob_spam = 1.0 if pred == 1 else 0.0
            prob_ham = 1.0 - prob_spam

        is_spam = bool(prob_spam >= 0.5)
        label = "SPAM" if is_spam else "HAM"
        confidence = prob_spam if is_spam else prob_ham

        # Identify which vocabulary tokens were found in this text
        feature_names = self.vectorizer.get_feature_names_out()
        nonzero_indices = X_vec.nonzero()[1]
        matched_tokens = [feature_names[i] for i in nonzero_indices]

        return {
            "label": label,
            "is_spam": is_spam,
            "confidence": float(confidence),
            "spam_probability": float(prob_spam),
            "ham_probability": float(prob_ham),
            "cleaned_text": cleaned,
            "matched_features": matched_tokens,
        }


def format_prediction_output(raw_text: str, result: dict) -> str:
    """Format the prediction result into a clear, aesthetic string."""
    tag = "[SPAM]" if result["is_spam"] else "[HAM]"
    conf_pct = result["confidence"] * 100
    spam_pct = result["spam_probability"] * 100
    ham_pct = result["ham_probability"] * 100

    banner = "=" * 60
    output = [
        banner,
        f"Input Message : \"{raw_text.strip()}\"",
        f"Cleaned Text  : \"{result['cleaned_text']}\"",
        f"Prediction    : {tag} ({'Spam detected' if result['is_spam'] else 'Legitimate message'})",
        f"Confidence    : {conf_pct:.2f}%",
        f"Probabilities : Spam: {spam_pct:.2f}% | Ham: {ham_pct:.2f}%",
    ]
    if result["matched_features"]:
        top_tokens = ", ".join(result["matched_features"][:8])
        output.append(f"Key Features  : {top_tokens}")
    output.append(banner)
    return "\n".join(output)


def interactive_session(predictor: SpamPredictor) -> None:
    """Run an interactive CLI loop where users can test messages."""
    print("\n" + "=" * 60)
    print("      SMS & EMAIL SPAM CLASSIFIER - INTERACTIVE MODE")
    print("=" * 60)
    print("Type or paste any message to test. Enter 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("Enter message > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting interactive mode. Goodbye!")
            break

        res = predictor.predict_one(user_input)
        print("\n" + format_prediction_output(user_input, res) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Classify messages as SPAM or HAM using trained ML models.")
    parser.add_argument("--text", "-t", type=str, help="Message text to classify")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive testing CLI")
    args = parser.parse_args()

    predictor = SpamPredictor()

    if args.text:
        res = predictor.predict_one(args.text)
        print(format_prediction_output(args.text, res))
    elif args.interactive or sys.stdin.isatty():
        interactive_session(predictor)
    else:
        # Read from piped stdin if available
        piped = sys.stdin.read().strip()
        if piped:
            res = predictor.predict_one(piped)
            print(format_prediction_output(piped, res))
        else:
            interactive_session(predictor)


if __name__ == "__main__":
    main()
