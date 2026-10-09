# Email & SMS Spam Classifier

An end-to-end Machine Learning system in Python for classifying SMS and Email messages as **HAM** (legitimate) or **SPAM** (unsolicited / scam / phishing). The project compares **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear Support Vector Classifier (LinearSVC)** using 5-fold Stratified Cross-Validation, with special focus on **high precision on spam** to minimize false positives.

---

## 📁 Project Structure

```
spam_classifier/
├── data/
│   └── spam.csv                    # UCI SMS Spam Collection dataset
├── src/
│   ├── preprocess.py               # Data loading, cleaning, and NLTK pipeline
│   ├── train.py                    # 5-fold CV training, evaluation, model saving
│   ├── predict.py                  # CLI & interactive real-time prediction
│   └── visualize.py                # Publication-quality charts & plots
├── models/
│   ├── best_model.joblib           # Serialized top-performing classifier (MultinomialNB)
│   ├── tfidf_vectorizer.joblib     # Fitted TF-IDF feature extractor (5,000 features)
│   └── model_metadata.json         # Performance metrics, hyperparameters, and vocab stats
├── outputs/
│   ├── class_distribution.png      # EDA class imbalance visualization
│   ├── message_length_distribution.png # Character and word length distributions
│   ├── confusion_matrices.png      # Side-by-side confusion matrices for all models
│   ├── model_comparison.png        # Bar chart comparing Accuracy, Precision, Recall, F1, AUC
│   ├── roc_curves.png              # Receiver Operating Characteristic curves
│   ├── top_spam_words.png          # Top 20 spam-indicative n-grams
│   └── metrics_summary.csv         # Detailed evaluation metrics table
├── notebook.ipynb                  # Step-by-step interactive walkthrough
├── requirements.txt                # Project dependencies
└── README.md                       # Documentation & setup guide
```

---

## 📊 Dataset Details & Ingestion

The model is trained on the **UCI SMS Spam Collection**:
- **Total records**: 5,572 messages (`4,825 ham`, `747 spam`).
- **Encoding**: Must be ingested using `encoding='latin-1'`.
- **Columns cleaned**: Original columns `v1` and `v2` renamed to `label` and `text`. Dropped 3 empty columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`).
- **Label Mapping**: `ham` $\rightarrow$ `0` (legitimate), `spam` $\rightarrow$ `1` (spam).
- **Deduplication**: 403 duplicate messages removed, leaving **5,169 unique samples** (`4,516 ham` [87.4%], `653 spam` [12.6%]).
- **Class Imbalance**: Handled via Stratified K-Fold splitting and balanced class weighting where appropriate.

---

## ⚙️ Text Preprocessing Pipeline (`src/preprocess.py`)

1. **Case Normalization**: All text converted to lowercase.
2. **Noise Reduction**: Strips URLs (`http://...`, `www...`), email addresses, and numerical digits.
3. **Punctuation & Special Character Removal**: Eliminates symbols while retaining word structure.
4. **Tokenization**: Words extracted via NLTK `word_tokenize`.
5. **Stopwords Elimination**: Filters out common English stopwords (`the`, `is`, `at`, etc.) from NLTK.
6. **Stemming**: Applies NLTK `PorterStemmer` to reduce tokens to their root stems (e.g., `winning` $\rightarrow$ `win`, `claimed` $\rightarrow$ `claim`).
7. **Leakage Protection**: All TF-IDF vocabulary extraction is strictly fit **only** on the 80% training set.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.9+ (tested on Python 3.10 – 3.14 on Windows)

### 2. Create Virtual Environment & Install Dependencies
Open your PowerShell terminal in the project directory:

```powershell
# Create virtual environment
py -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## 🏃 Usage

### 1. Run Complete Training Pipeline
Trains all 3 models with 5-fold Stratified GridSearchCV, evaluates on test data, generates all visualizations in `outputs/`, and saves the best model to `models/`:

```powershell
py src/train.py
```

### 2. Real-Time Inference (CLI)
Classify any message directly from the command line:

```powershell
py src/predict.py --text "Congratulations! You won a $1,000 Walmart card. Call 08001234 now!"
```

**Output:**
```
============================================================
Input Message : "Congratulations! You won a $1,000 Walmart card. Call 08001234 now!"
Cleaned Text  : "congratul walmart card call claim prize"
Prediction    : [SPAM] (Spam detected)
Confidence    : 99.92%
Probabilities : Spam: 99.92% | Ham: 0.08%
Key Features  : call, card, congratul, prize
============================================================
```

```powershell
py src/predict.py --text "Hey, are you free for lunch tomorrow around 1pm?"
```

**Output:**
```
============================================================
Input Message : "Hey, are you free for lunch tomorrow around 1pm?"
Cleaned Text  : "hey free lunch tomorrow around pm let know"
Prediction    : [HAM] (Legitimate message)
Confidence    : 99.85%
Probabilities : Spam: 0.15% | Ham: 99.85%
Key Features  : around, free, hey, know, let, lunch, pm
============================================================
```

### 3. Interactive Shell Mode
Launch an interactive session to test arbitrary messages:

```powershell
py src/predict.py --interactive
```

---

## 📈 Experimental Results & Model Comparison

Evaluated on an independent, held-out stratified test set (1,034 samples: 903 Ham, 131 Spam):

| Model | Hyperparameters | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) | ROC-AUC | False Positives | FPR (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** ⭐ | `alpha=0.05` | **98.26%** | **97.48%** | 88.55% | **0.9280** | **0.9927** | **3** | **0.33%** |
| **Linear SVM (LinearSVC)** | `C=0.5, balanced` | 98.16% | 93.75% | **91.60%** | 0.9266 | 0.9859 | 8 | 0.89% |
| **Logistic Regression** | `C=10.0, liblinear, balanced` | 97.97% | 92.97% | 90.84% | 0.9189 | 0.9888 | 9 | 1.00% |

### Why Multinomial Naive Bayes Performed Best
In spam detection systems, the cost of a **False Positive** (a genuine, important email sent to Spam or deleted) is vastly higher than a **False Negative** (an annoying spam message arriving in the inbox).
- **Lowest False Positive Rate (0.33%)**: Multinomial Naive Bayes produced only **3 false positives** across 903 legitimate messages (compared to 8 for LinearSVC and 9 for Logistic Regression).
- **Highest Precision (97.48%)**: Over 97.4% of messages flagged as spam by Naive Bayes were truly spam.
- **Top ROC-AUC (0.9927)**: Strongest global separation between the positive and negative class distributions.
- **Independence Assumption Advantage**: Despite the strong feature conditional independence assumption, smoothed MultinomialNB (`alpha=0.05`) remains extremely resilient to class imbalance in sparse text feature spaces.

---

## 🔍 Top 20 Spam-Indicating Features

The top predictive tokens identified by the models include:
1. `claim`
2. `prize`
3. `won` / `winner`
4. `call` / `txt` / `text`
5. `free`
6. `servic`
7. `repli`
8. `urgent`
9. `custom`
10. `award`
11. `mobil`
12. `contact`
13. `guarante`
14. `cash`
15. `tone`
16. `select`
17. `custom servic`
18. `messag`
19. `stop`
20. `per week`

---

## ⚠️ Limitations & Future Work

1. **SMS vs. Full-Body Email Discrepancy**:
   - The UCI dataset consists exclusively of SMS text messages. SMS messages are character-constrained (<160 chars) and heavily rely on phone numbers, shorthand (`txt`, `u`, `ur`), and call-to-actions.
   - Real-world emails contain multi-paragraph text, HTML templates, CSS styles, and rich attachments.
2. **Lack of Metadata & Headers**:
   - Modern spam filters rely heavily on email headers (DKIM, SPF, DMARC validation, sender IP reputation, routing hops). Text-only classification cannot detect spoofed domains or header tampering.
3. **Adversarial Drift (Spam Evolution)**:
   - Modern spammers actively bypass keyword filters using homoglyphs, zero-width spaces, and image-based spam. Future enhancements can incorporate Subword / BPE tokenization or fine-tuned transformer embeddings (e.g., DistilBERT or DeBERTa).
