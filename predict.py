"""
Prediction and Inference Pipeline with Keyword Explainability
==============================================================
Project: Fake News Detection System Using Python and Machine Learning
Target Audience: 3rd-year B.Sc. Artificial Intelligence and Machine Learning

This module provides real-time inference functionality:
1. Loads the serialized model and TF-IDF vectorizer using Joblib with caching.
2. Validates user input to prevent empty or garbage inferences.
3. Preprocesses input text consistently with the training pipeline.
4. Generates predictions ("Likely Reliable" or "Potentially Fake").
5. Computes model confidence scores (calibrated for both Logistic Regression and Passive-Aggressive).
6. Performs transparent feature contribution analysis (identifying key n-grams influencing the prediction).
"""

import os
import sys
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import clean_text, combine_content, decode_label, validate_input_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "news_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.joblib")
LR_PATH = os.path.join(MODELS_DIR, "logistic_regression.joblib")
PAC_PATH = os.path.join(MODELS_DIR, "passive_aggressive.joblib")

# Global in-memory cache for fast repeated predictions
_CACHED_MODEL = None
_CACHED_VECTORIZER = None
_CACHED_METRICS = None
_CACHED_LR = None
_CACHED_PAC = None


def is_model_trained() -> bool:
    """
    Checks whether the trained model, vectorizer, and metrics files exist on disk.
    """
    return os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(METRICS_PATH)


def load_artifacts():
    """
    Loads saved model, vectorizer, and metadata from disk with caching.
    """
    global _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_METRICS, _CACHED_LR, _CACHED_PAC

    if not is_model_trained():
        raise FileNotFoundError(
            "Trained model artifacts were not found in 'models/' directory. "
            "Please run 'python train_model.py' to train and save the models."
        )

    if _CACHED_MODEL is None:
        _CACHED_MODEL = joblib.load(MODEL_PATH)
        _CACHED_VECTORIZER = joblib.load(VECTORIZER_PATH)
        _CACHED_METRICS = joblib.load(METRICS_PATH)
        if os.path.exists(LR_PATH):
            _CACHED_LR = joblib.load(LR_PATH)
        if os.path.exists(PAC_PATH):
            _CACHED_PAC = joblib.load(PAC_PATH)

    return _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_METRICS


def explain_prediction(text_vector, feature_names, model_coef, top_n=6):
    """
    Extracts the most influential TF-IDF n-grams from the user's input text based on model weights.

    Parameters
    ----------
    text_vector : scipy.sparse matrix
        TF-IDF vector of the input text (1 x num_features).
    feature_names : list or np.ndarray
        Array of feature token strings from TfidfVectorizer.
    model_coef : np.ndarray
        Model feature weights / coefficients (1 x num_features).
    top_n : int
        Number of top features to return.

    Returns
    -------
    list of dict
        Tokens found in text, sorted by absolute weight impact.
    """
    non_zero_indices = text_vector.nonzero()[1]
    if len(non_zero_indices) == 0:
        return []

    contributions = []
    for idx in non_zero_indices:
        token = feature_names[idx]
        tfidf_val = text_vector[0, idx]
        weight = model_coef[0, idx]
        impact = tfidf_val * weight

        lean = "Fake" if weight > 0 else "Reliable"
        contributions.append({
            "token": token,
            "impact": float(impact),
            "weight": float(weight),
            "lean": lean
        })

    # Sort by absolute impact magnitude
    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return contributions[:top_n]


def predict_news(headline: str, article: str, model_choice: str = "Best Model") -> dict:
    """
    Classifies a news article and calculates prediction confidence and contributing features.

    Parameters
    ----------
    headline : str
        News headline.
    article : str
        News article body.
    model_choice : str
        Which model to use: "Best Model", "Logistic Regression", or "Passive-Aggressive Classifier".

    Returns
    -------
    dict
        Structured prediction response dictionary.
    """
    # 1. Validate inputs
    is_valid, error_msg = validate_input_text(headline, article)
    if not is_valid:
        return {
            "status": "error",
            "message": error_msg
        }

    # 2. Check if models are available
    if not is_model_trained():
        return {
            "status": "error",
            "message": (
                "The machine learning model has not been trained yet. "
                "Please run `python train_model.py` in your terminal or train the model "
                "before analyzing articles."
            )
        }

    try:
        best_model, vectorizer, metrics_data = load_artifacts()

        # Select model according to user choice
        if model_choice == "Logistic Regression" and _CACHED_LR is not None:
            active_model = _CACHED_LR
            model_name = "Logistic Regression"
        elif model_choice == "Passive-Aggressive Classifier" and _CACHED_PAC is not None:
            active_model = _CACHED_PAC
            model_name = "Passive-Aggressive Classifier"
        else:
            active_model = best_model
            model_name = metrics_data.get("best_model_name", "Passive-Aggressive Classifier")

        # 3. Combine and clean text
        raw_combined = combine_content(headline, article)
        cleaned = clean_text(raw_combined)
        word_count = len(raw_combined.split())

        if not cleaned:
            return {
                "status": "error",
                "message": "Input contains only punctuation, URLs, or special characters. Please provide descriptive text."
            }

        # 4. Vectorize text using fitted TF-IDF
        text_vector = vectorizer.transform([cleaned])

        # 5. Predict label
        pred_class = int(active_model.predict(text_vector)[0])
        pred_label = decode_label(pred_class)

        # 6. Compute Confidence Score (%)
        confidence = 50.0
        if hasattr(active_model, "predict_proba"):
            # Probabilities available (e.g. Logistic Regression)
            proba = active_model.predict_proba(text_vector)[0]
            confidence = float(max(proba) * 100.0)
        elif hasattr(active_model, "decision_function"):
            # Calibrate decision boundary using sigmoid function: P = 1 / (1 + exp(-decision))
            decision = float(active_model.decision_function(text_vector)[0])
            prob_fake = 1.0 / (1.0 + np.exp(-decision))
            confidence = float(prob_fake * 100.0 if pred_class == 1 else (1.0 - prob_fake) * 100.0)

        # Clamp confidence to realistic 51.0% - 99.5%
        confidence = round(max(51.0, min(99.5, confidence)), 1)

        # 7. Extract contributing informative tokens
        feature_names = vectorizer.get_feature_names_out()
        model_coef = active_model.coef_
        key_tokens = explain_prediction(text_vector, feature_names, model_coef, top_n=6)

        # 8. Generate plain-English explanation
        if pred_class == 1:
            explanation = (
                "The text exhibits linguistic patterns, keywords, or stylistic cues commonly found in "
                "sensationalized, conspiratorial, or fabricated news records in the training dataset."
            )
        else:
            explanation = (
                "The text aligns with the formal structure, neutral vocabulary, and institutional phrasing "
                "characteristic of standard journalistic reporting in the training dataset."
            )

        return {
            "status": "success",
            "prediction_label": pred_label,
            "prediction_int": pred_class,
            "confidence": confidence,
            "model_name": model_name,
            "word_count": word_count,
            "key_tokens": key_tokens,
            "explanation": explanation,
            "headline": headline,
            "article_preview": (article[:160] + "...") if len(article) > 160 else article
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"An unexpected error occurred during prediction: {str(e)}"
        }
