"""Training, evaluation, and model selection pipeline for SMS/Email Spam Classifier.

Models evaluated:
1. Multinomial Naive Bayes (MultinomialNB)
2. Logistic Regression (balanced class weight)
3. Support Vector Classifier (LinearSVC with balanced class weight)

Features:
- Stratified 80/20 train-test split (fixed random_state=42)
- TF-IDF unigram + bigram vectorizer (max_features=5000, fit ONLY on training set)
- 5-fold Stratified GridSearchCV hyperparameter tuning
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, False Positive Rate
- Visualizations: EDA, Confusion Matrices, ROC Curves, Performance Comparison, Top Spam Words
- Serializes best model and vectorizer to /models
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

# Support running directly or as a package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import load_and_clean_data, download_nltk_resources
from src.visualize import (
    plot_class_distribution,
    plot_message_length_distribution,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_top_spam_words,
    plot_roc_curves,
)


def run_pipeline(
    data_path: str = None,
    output_dir: str = None,
    models_dir: str = None,
    random_state: int = 42
) -> dict:
    """Executes the full end-to-end training and evaluation pipeline."""
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, "data", "spam.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(PROJECT_ROOT, "spam.csv")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "outputs")
    if models_dir is None:
        models_dir = os.path.join(PROJECT_ROOT, "models")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70, flush=True)
    print("      EMAIL & SMS SPAM CLASSIFIER - TRAINING & EVALUATION PIPELINE", flush=True)
    print("=" * 70, flush=True)

    # 1. Download NLTK datasets
    download_nltk_resources()

    # 2. Load & Clean Data
    print(f"\n[Step 1/7] Loading and cleaning dataset from: {data_path}", flush=True)
    df = load_and_clean_data(data_path)

    # 3. Exploratory Data Analysis & Plots
    print("\n[Step 2/7] Generating EDA visualizations...", flush=True)
    class_dist_plot = os.path.join(output_dir, "class_distribution.png")
    length_dist_plot = os.path.join(output_dir, "message_length_distribution.png")
    plot_class_distribution(df, class_dist_plot)
    plot_message_length_distribution(df, length_dist_plot)

    # 4. Stratified Train / Test Split (80 / 20)
    print("\n[Step 3/7] Performing Stratified 80/20 Train-Test Split...", flush=True)
    X = df["cleaned_text"]
    y = df["label"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    print(f"  Training samples: {len(X_train_raw)} (Ham: {(y_train == 0).sum()}, Spam: {(y_train == 1).sum()})", flush=True)
    print(f"  Testing samples:  {len(X_test_raw)} (Ham: {(y_test == 0).sum()}, Spam: {(y_test == 1).sum()})", flush=True)

    # 5. Feature Extraction (TF-IDF Vectorizer) - Fit ONLY on Train!
    print("\n[Step 4/7] Fitting TF-IDF Vectorizer (unigrams + bigrams, max_features=5000)...", flush=True)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=2,
    )

    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)} features", flush=True)
    print(f"  X_train_vec shape: {X_train_vec.shape}, X_test_vec shape: {X_test_vec.shape}", flush=True)

    # 6. Model Training & 5-fold Stratified GridSearchCV Tuning
    print("\n[Step 5/7] Tuning hyperparameters with 5-Fold Stratified GridSearchCV...", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    model_configs = {
        "Multinomial Naive Bayes": {
            "estimator": MultinomialNB(),
            "param_grid": {
                "alpha": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            },
        },
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "param_grid": {
                "C": [0.1, 0.5, 1.0, 5.0, 10.0],
                "solver": ["liblinear", "lbfgs"],
            },
        },
        "Linear SVM": {
            "estimator": LinearSVC(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "param_grid": {
                "C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            },
        },
    }

    trained_models = {}
    best_params_dict = {}

    for name, config in model_configs.items():
        print(f"  - Optimizing {name}...", flush=True)
        grid = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            cv=cv,
            scoring="f1",
            n_jobs=1,
        )
        grid.fit(X_train_vec, y_train)
        trained_models[name] = grid.best_estimator_
        best_params_dict[name] = grid.best_params_
        print(f"    Best params: {grid.best_params_} (CV F1 Score: {grid.best_score_:.4f})", flush=True)

    # 7. Comprehensive Evaluation on Test Set
    print("\n[Step 6/7] Evaluating all candidate models on held-out test data...", flush=True)
    metrics_records = []
    cm_dict = {}
    roc_dict = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test_vec)

        # Probabilities or decision scores for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test_vec)[:, 1]
        else:
            y_scores = model.decision_function(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        auc = roc_auc_score(y_test, y_scores)
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        cm_dict[name] = cm
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_scores)
        roc_dict[name] = (fpr_curve, tpr_curve, auc)

        metrics_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "True Negatives": int(tn),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "True Positives": int(tp),
            "FPR": fpr,
            "Best Parameters": str(best_params_dict[name]),
        })

        print(f"\n--- Classification Report: {name} ---", flush=True)
        print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"], digits=4), flush=True)
        print(f"False Positives: {fp} / {tn + fp} legitimate messages (FPR: {fpr * 100:.2f}%)", flush=True)

    metrics_df = pd.DataFrame(metrics_records)

    # Save metrics table
    csv_metrics_path = os.path.join(output_dir, "metrics_summary.csv")
    metrics_df.to_csv(csv_metrics_path, index=False)
    print(f"\n[Metrics] Saved test evaluation summary to: {csv_metrics_path}", flush=True)

    # Generate visual outputs
    cm_plot_path = os.path.join(output_dir, "confusion_matrices.png")
    comp_plot_path = os.path.join(output_dir, "model_comparison.png")
    roc_plot_path = os.path.join(output_dir, "roc_curves.png")
    top_words_path = os.path.join(output_dir, "top_spam_words.png")

    plot_confusion_matrices(cm_dict, cm_plot_path)
    plot_model_comparison(metrics_df, comp_plot_path)
    plot_roc_curves(roc_dict, roc_plot_path)

    # Feature Importance (from Logistic Regression or Naive Bayes)
    feature_names = np.array(vectorizer.get_feature_names_out())
    lr_model = trained_models["Logistic Regression"]
    coefs = lr_model.coef_[0]
    top_spam_indices = np.argsort(coefs)[::-1][:20]
    top_spam_features = feature_names[top_spam_indices]
    top_spam_weights = coefs[top_spam_indices]

    plot_top_spam_words(
        list(top_spam_features),
        list(top_spam_weights),
        top_words_path,
        title="Top 20 Spam-Indicating Words & Bigrams (Logistic Regression)",
    )

    # 8. Selection and Serialization of Best Model
    print("\n[Step 7/7] Selecting and saving best model...", flush=True)

    # Prioritize Precision (minimum False Positives) then F1-score
    ranked_df = metrics_df.sort_values(
        by=["Precision", "F1-Score", "ROC-AUC"], ascending=[False, False, False]
    ).reset_index(drop=True)

    best_model_name = ranked_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_stats = ranked_df.iloc[0].to_dict()

    print(f"\n>>> BEST PERFORMING MODEL: {best_model_name} <<<", flush=True)
    print(f"    Precision (Spam):  {best_stats['Precision']:.4f}", flush=True)
    print(f"    Recall (Spam):     {best_stats['Recall']:.4f}", flush=True)
    print(f"    F1-Score (Spam):   {best_stats['F1-Score']:.4f}", flush=True)
    print(f"    Accuracy:          {best_stats['Accuracy']:.4f}", flush=True)
    print(f"    False Positives:   {best_stats['False Positives']} (FPR: {best_stats['FPR'] * 100:.2f}%)", flush=True)
    print(f"    ROC-AUC:           {best_stats['ROC-AUC']:.4f}", flush=True)

    # Serialize best model and vectorizer
    model_save_path = os.path.join(models_dir, "best_model.joblib")
    vectorizer_save_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    meta_save_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(best_model, model_save_path)
    joblib.dump(vectorizer, vectorizer_save_path)

    metadata = {
        "best_model_name": best_model_name,
        "best_hyperparameters": best_params_dict[best_model_name],
        "metrics": {
            "accuracy": float(best_stats["Accuracy"]),
            "precision": float(best_stats["Precision"]),
            "recall": float(best_stats["Recall"]),
            "f1_score": float(best_stats["F1-Score"]),
            "roc_auc": float(best_stats["ROC-AUC"]),
            "false_positives": int(best_stats["False Positives"]),
            "false_negatives": int(best_stats["False Negatives"]),
            "true_positives": int(best_stats["True Positives"]),
            "true_negatives": int(best_stats["True Negatives"]),
            "fpr": float(best_stats["FPR"]),
        },
        "all_models_summary": metrics_df.to_dict(orient="records"),
        "vocab_size": len(vectorizer.vocabulary_),
    }

    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSuccessfully saved:", flush=True)
    print(f"  - Model:      {model_save_path}", flush=True)
    print(f"  - Vectorizer: {vectorizer_save_path}", flush=True)
    print(f"  - Metadata:   {meta_save_path}", flush=True)

    return {
        "metrics_df": metrics_df,
        "best_model_name": best_model_name,
        "best_stats": best_stats,
    }


if __name__ == "__main__":
    run_pipeline()
