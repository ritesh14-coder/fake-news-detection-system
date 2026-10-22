"""
Analysis History Logger and Storage Manager
============================================
Project: Fake News Detection System Using Python and Machine Learning
Target Audience: 3rd-year B.Sc. Artificial Intelligence and Machine Learning

This module manages the local logging of user analysis records:
- Appends prediction results to `history/analysis_history.csv`.
- Retrieves history records as Pandas DataFrames for Streamlit display.
- Calculates summary statistics (total queries, fake vs reliable distribution).
- Supports clearing analysis history safely.
"""

import os
import datetime
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "analysis_history.csv")

HISTORY_COLUMNS = [
    "Timestamp",
    "Headline",
    "Text_Preview",
    "Prediction",
    "Confidence",
    "Model_Used",
    "Word_Count"
]


def ensure_history_file():
    """
    Ensures that the history directory and CSV file exist with appropriate headers.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        df_empty = pd.DataFrame(columns=HISTORY_COLUMNS)
        df_empty.to_csv(HISTORY_FILE, index=False)


def log_prediction(
    headline: str,
    article: str,
    prediction: str,
    confidence: float,
    model_used: str = "Passive-Aggressive Classifier",
    word_count: int = 0
) -> bool:
    """
    Appends a new detection record to the CSV history log.

    Parameters
    ----------
    headline : str
        News article headline.
    article : str
        Full article body or text preview.
    prediction : str
        Predicted label ('Likely Reliable' or 'Potentially Fake').
    confidence : float
        Model confidence percentage (e.g. 94.5).
    model_used : str
        Name of the model used for inference.
    word_count : int
        Number of words in submitted text.

    Returns
    -------
    bool
        True if successfully logged, False otherwise.
    """
    try:
        ensure_history_file()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        headline_clean = str(headline).strip() if headline else "No Headline Provided"
        
        # Keep shortened preview of article to keep history clean and compact
        article_str = str(article).strip() if article else ""
        if len(article_str) > 150:
            preview = article_str[:150] + "..."
        else:
            preview = article_str if article_str else "N/A"

        new_record = pd.DataFrame([{
            "Timestamp": timestamp,
            "Headline": headline_clean,
            "Text_Preview": preview,
            "Prediction": prediction,
            "Confidence": round(float(confidence), 1),
            "Model_Used": model_used,
            "Word_Count": int(word_count)
        }])

        new_record.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
        return True
    except Exception as e:
        print(f"[Warning] Failed to log history entry: {e}")
        return False


def load_history() -> pd.DataFrame:
    """
    Loads analysis history from CSV as a Pandas DataFrame.
    """
    ensure_history_file()
    try:
        df = pd.read_csv(HISTORY_FILE)
        if df.empty:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        # Sort newest first
        if "Timestamp" in df.columns:
            df["Timestamp_Parsed"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.sort_values(by="Timestamp_Parsed", ascending=False).drop(columns=["Timestamp_Parsed"])
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def clear_history() -> bool:
    """
    Clears all saved records in the history log while preserving the schema header.
    """
    try:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        df_empty = pd.DataFrame(columns=HISTORY_COLUMNS)
        df_empty.to_csv(HISTORY_FILE, index=False)
        return True
    except Exception as e:
        print(f"[Warning] Failed to clear history: {e}")
        return False


def get_history_stats() -> dict:
    """
    Calculates aggregate metrics on past user analyses.
    """
    df = load_history()
    if df.empty:
        return {
            "total_queries": 0,
            "reliable_count": 0,
            "fake_count": 0,
            "avg_confidence": 0.0
        }

    total = len(df)
    reliable = int(np.sum(df["Prediction"] == "Likely Reliable")) if "Prediction" in df.columns else 0
    fake = int(np.sum(df["Prediction"] == "Potentially Fake")) if "Prediction" in df.columns else 0
    avg_conf = float(df["Confidence"].mean()) if "Confidence" in df.columns and not df["Confidence"].isna().all() else 0.0

    return {
        "total_queries": total,
        "reliable_count": reliable,
        "fake_count": fake,
        "avg_confidence": round(avg_conf, 1)
    }
