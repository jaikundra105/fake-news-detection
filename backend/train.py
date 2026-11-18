from pathlib import Path
import re
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "news.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
    return " ".join(tokens)

def normalize_label(value):
    value = str(value).strip().lower()
    if value in {"1", "real", "true", "reliable"}:
        return 1
    if value in {"0", "fake", "false", "unreliable"}:
        return 0
    raise ValueError(f"Unrecognized label: {value}")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Add backend/data/news.csv with columns text,label. "
            "Labels can be REAL/FAKE or 1/0."
        )

    df = pd.read_csv(DATA_PATH)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns named 'text' and 'label'.")
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 20]
    df["label"] = df["label"].apply(normalize_label)
    df = df.drop_duplicates(subset=["text"])

    if df["label"].nunique() != 2 or df["label"].value_counts().min() < 2:
        raise ValueError("Need at least two examples from each class.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    models = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced")
    }

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=clean_text,
                ngram_range=(1, 2),
                max_features=50000,
                min_df=2,
                sublinear_tf=True
            )),
            ("classifier", classifier)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        print(f"\\n===== {name.upper()} =====")
        print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
        print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
        print("Recall   :", round(recall_score(y_test, predictions, zero_division=0), 4))
        print("F1-score :", round(f1_score(y_test, predictions, zero_division=0), 4))
        print(classification_report(y_test, predictions, target_names=["FAKE", "REAL"], zero_division=0))
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")

    print("\\nTraining complete. Model files saved in backend/models/")

if __name__ == "__main__":
    main()
