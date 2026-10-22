"""
Machine Learning Model Training Pipeline
=========================================
Project: Fake News Detection System Using Python and Machine Learning
Target Audience: 3rd-year B.Sc. Artificial Intelligence and Machine Learning

This script implements the full end-to-end Machine Learning training pipeline:
1. Loads the labeled dataset from CSV using Pandas.
2. Cleans text, handles missing values, and drops duplicates to prevent data leakage.
3. Combines headlines and article text into composite text features.
4. Performs a stratified train-test split (80% train, 20% test, random_state=42).
5. Fits a TfidfVectorizer strictly on training data to prevent lookahead bias / data leakage.
6. Trains two competitive classification models:
   - Logistic Regression (interpretable, well-calibrated probabilistic baseline)
   - Passive-Aggressive Classifier (online, large-margin classifier suited for streaming news)
7. Evaluates both models on the held-out test set:
   - Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
8. Persists the best model, vectorizer, full pipeline, and metrics dictionary using Joblib.
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline

# Import utility preprocessing routines
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import clean_text, combine_content, encode_label, LABEL_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "news_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_and_preprocess_data(csv_path: str):
    """
    Loads dataset, handles null values, deduplicates, and cleans text.

    Parameters
    ----------
    csv_path : str
        Path to the news CSV dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned, deduplicated DataFrame with 'clean_text' and 'encoded_label' columns.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}. Please generate or add the dataset.")

    print(f"[*] Step 1: Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    print(f"    Loaded {initial_rows} raw records with columns: {list(df.columns)}")

    # 1. Fill missing values
    if "title" in df.columns:
        df["title"] = df["title"].fillna("")
    else:
        df["title"] = ""

    if "text" in df.columns:
        df["text"] = df["text"].fillna("")
    else:
        df["text"] = ""

    # Ensure label column exists
    if "label" not in df.columns:
        possible_label_cols = [c for c in df.columns if "label" in c.lower() or "target" in c.lower() or "class" in c.lower()]
        if possible_label_cols:
            df["label"] = df[possible_label_cols[0]]
        else:
            raise KeyError("Dataset must contain a 'label' column.")

    # 2. Standardize target labels
    df["encoded_label"] = df["label"].apply(encode_label)

    # 3. Combine title and text
    df["combined_text"] = df.apply(lambda row: combine_content(row["title"], row["text"]), axis=1)

    # 4. Clean text using preprocessing function
    print("[*] Step 2: Preprocessing and normalizing text...")
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # 5. Remove empty text records
    df = df[df["clean_text"].str.len() > 10].copy()

    # 6. Remove duplicate records to prevent data leakage between train and test sets
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    deduped_rows = len(df)
    dropped = initial_rows - deduped_rows
    if dropped > 0:
        print(f"    Dropped {dropped} duplicate/unusable records to prevent data leakage.")
    print(f"    Final usable dataset: {deduped_rows} records.")

    label_counts = df["encoded_label"].value_counts().to_dict()
    print(f"    Class Balance: {label_counts.get(0, 0)} Reliable (0), {label_counts.get(1, 0)} Fake (1)")

    return df


def train_and_evaluate(df: pd.DataFrame):
    """
    Splits data, fits TF-IDF vectorizer, trains models, and calculates test metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.

    Returns
    -------
    dict
        Dictionary containing trained models, vectorizer, and evaluation metrics.
    """
    # Convert to standard NumPy array to ensure full compatibility with scikit-learn
    X = np.array(df["clean_text"].tolist(), dtype=object)
    y = np.array(df["encoded_label"].tolist(), dtype=int)


    print("\n[*] Step 3: Performing Stratified Train-Test Split (80% Train, 20% Test, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"    Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # Step 4: TF-IDF Vectorization
    # CRITICAL ACADEMIC NOTE: Fit vectorizer strictly on training set to avoid lookahead bias / data leakage.
    print("\n[*] Step 4: Extracting TF-IDF Features (unigrams + bigrams, max_features=5000, English stop words)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english",
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"    Vocabulary Size: {vocab_size} distinct n-grams.")

    # Step 5: Model 1 - Logistic Regression
    print("\n[*] Step 5: Training Model 1: Logistic Regression...")
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    lr_preds = lr_model.predict(X_test_tfidf)

    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds, zero_division=0)
    lr_rec = recall_score(y_test, lr_preds, zero_division=0)
    lr_f1 = f1_score(y_test, lr_preds, zero_division=0)
    lr_cm = confusion_matrix(y_test, lr_preds)

    print(f"    Logistic Regression -> Acc: {lr_acc:.4f} | Prec: {lr_prec:.4f} | Rec: {lr_rec:.4f} | F1: {lr_f1:.4f}")

    # Step 6: Model 2 - Passive-Aggressive Classifier
    print("\n[*] Step 6: Training Model 2: Passive-Aggressive Classifier...")
    pac_model = PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5)
    pac_model.fit(X_train_tfidf, y_train)
    pac_preds = pac_model.predict(X_test_tfidf)

    pac_acc = accuracy_score(y_test, pac_preds)
    pac_prec = precision_score(y_test, pac_preds, zero_division=0)
    pac_rec = recall_score(y_test, pac_preds, zero_division=0)
    pac_f1 = f1_score(y_test, pac_preds, zero_division=0)
    pac_cm = confusion_matrix(y_test, pac_preds)

    print(f"    Passive-Aggressive  -> Acc: {pac_acc:.4f} | Prec: {pac_prec:.4f} | Rec: {pac_rec:.4f} | F1: {pac_f1:.4f}")

    # Model comparison table
    comparison_df = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Accuracy": round(lr_acc, 4),
            "Precision": round(lr_prec, 4),
            "Recall": round(lr_rec, 4),
            "F1_Score": round(lr_f1, 4)
        },
        {
            "Model": "Passive-Aggressive Classifier",
            "Accuracy": round(pac_acc, 4),
            "Precision": round(pac_prec, 4),
            "Recall": round(pac_rec, 4),
            "F1_Score": round(pac_f1, 4)
        }
    ])

    print("\n=== MODEL PERFORMANCE COMPARISON (HELD-OUT TEST SET) ===")
    print(comparison_df.to_string(index=False))

    # Select best model based on F1-score (harmonic mean of precision and recall)
    if pac_f1 >= lr_f1:
        best_name = "Passive-Aggressive Classifier"
        best_model = pac_model
        best_preds = pac_preds
        best_acc, best_prec, best_rec, best_f1, best_cm = pac_acc, pac_prec, pac_rec, pac_f1, pac_cm
    else:
        best_name = "Logistic Regression"
        best_model = lr_model
        best_preds = lr_preds
        best_acc, best_prec, best_rec, best_f1, best_cm = lr_acc, lr_prec, lr_rec, lr_f1, lr_cm

    print(f"\n[+] Best Selected Model: {best_name} (F1 = {best_f1:.4f})")

    # Classification report
    target_names = ["Likely Reliable (0)", "Potentially Fake (1)"]
    report_dict = classification_report(y_test, best_preds, target_names=target_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, best_preds, target_names=target_names, zero_division=0)
    print("\nClassification Report:")
    print(report_text)

    # Build a unified pipeline for production deployment
    best_pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", best_model)
    ])

    results = {
        "best_model": best_model,
        "best_name": best_name,
        "best_pipeline": best_pipeline,
        "vectorizer": vectorizer,
        "lr_model": lr_model,
        "pac_model": pac_model,
        "comparison_df": comparison_df,
        "test_metrics": {
            "accuracy": float(best_acc),
            "precision": float(best_prec),
            "recall": float(best_rec),
            "f1_score": float(best_f1),
            "confusion_matrix": best_cm.tolist(),
            "classification_report_dict": report_dict,
            "classification_report_text": report_text,
        },
        "lr_metrics": {
            "accuracy": float(lr_acc),
            "precision": float(lr_prec),
            "recall": float(lr_rec),
            "f1_score": float(lr_f1),
            "confusion_matrix": lr_cm.tolist(),
        },
        "pac_metrics": {
            "accuracy": float(pac_acc),
            "precision": float(pac_prec),
            "recall": float(pac_rec),
            "f1_score": float(pac_f1),
            "confusion_matrix": pac_cm.tolist(),
        },
        "dataset_stats": {
            "total_records": len(df),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "vocab_size": vocab_size,
            "reliable_count": int(np.sum(y == 0)),
            "fake_count": int(np.sum(y == 1)),
            "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    return results


def save_artifacts(results: dict):
    """
    Saves trained models, vectorizer, and evaluation metrics to `models/` directory using Joblib.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"\n[*] Step 7: Persisting models and metrics to: {MODELS_DIR}...")

    # 1. Best model
    best_model_path = os.path.join(MODELS_DIR, "news_model.joblib")
    joblib.dump(results["best_model"], best_model_path)
    print(f"    Saved Best Model: {best_model_path}")

    # 2. TF-IDF vectorizer
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(results["vectorizer"], vectorizer_path)
    print(f"    Saved Vectorizer: {vectorizer_path}")

    # 3. Full Scikit-learn Pipeline
    pipeline_path = os.path.join(MODELS_DIR, "news_pipeline.joblib")
    joblib.dump(results["best_pipeline"], pipeline_path)
    print(f"    Saved Full Pipeline: {pipeline_path}")

    # 4. Save both individual models for interactive comparison
    lr_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    joblib.dump(results["lr_model"], lr_path)

    pac_path = os.path.join(MODELS_DIR, "passive_aggressive.joblib")
    joblib.dump(results["pac_model"], pac_path)

    # 5. Save comprehensive metrics and metadata
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.joblib")
    metrics_payload = {
        "best_model_name": results["best_name"],
        "comparison_df": results["comparison_df"],
        "test_metrics": results["test_metrics"],
        "lr_metrics": results["lr_metrics"],
        "pac_metrics": results["pac_metrics"],
        "dataset_stats": results["dataset_stats"]
    }
    joblib.dump(metrics_payload, metrics_path)
    print(f"    Saved Metrics & Evaluation Data: {metrics_path}")

    print("\n[SUCCESS] Machine learning training pipeline completed successfully!")


def main():
    print("=========================================================")
    print(" Fake News Detection System - Machine Learning Pipeline  ")
    print("=========================================================\n")
    df = load_and_preprocess_data(DATASET_PATH)
    results = train_and_evaluate(df)
    save_artifacts(results)


if __name__ == "__main__":
    main()
