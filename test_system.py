"""
Automated Test Suite for Fake News Detection System
===================================================
Project: Fake News Detection System Using Python and Machine Learning
Target Audience: 3rd-year B.Sc. Artificial Intelligence and Machine Learning

This script performs end-to-end verification of all system modules:
1. utils: text cleaning and input validation
2. dataset: schema integrity and class distribution
3. models: persistence and loading
4. predict: real-time classification, confidence calibration, and explainability
5. history_manager: CSV logging, retrieval, and clearing
6. data_analysis: summary statistics and matplotlib/seaborn chart generation
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from utils import clean_text, combine_content, encode_label, decode_label, validate_input_text
from predict import predict_news, is_model_trained, load_artifacts
from history_manager import log_prediction, load_history, clear_history, get_history_stats
from data_analysis import (
    load_dataset,
    get_dataset_summary,
    plot_label_distribution,
    plot_text_length_distribution,
    get_top_ngrams,
    search_dataset
)


class TestFakeNewsDetectionSystem(unittest.TestCase):

    def test_01_utils_clean_text(self):
        """Test text cleaning strips HTML, URLs, and punctuation."""
        raw = "BREAKING! Check out http://fake.com <p>Shocking Revelation!!!</p> 100% cure!"
        cleaned = clean_text(raw)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertIn("breaking", cleaned)
        self.assertIn("shocking", cleaned)
        self.assertIn("revelation", cleaned)
        self.assertIn("cure", cleaned)

    def test_02_utils_combine_and_validate(self):
        """Test combine content and input validation."""
        title = "NASA Mars Rover Discovery"
        body = "Scientists found organic compounds."
        combined = combine_content(title, body)
        self.assertIn("NASA Mars Rover Discovery", combined)
        self.assertIn("Scientists found organic compounds.", combined)

        # Validation checks
        valid, msg = validate_input_text("", "")
        self.assertFalse(valid)

        valid, msg = validate_input_text("Too short", "")
        self.assertFalse(valid)

        valid, msg = validate_input_text(title, body)
        self.assertTrue(valid)

    def test_03_dataset_integrity(self):
        """Test dataset exists, has required columns, and balanced classes."""
        csv_path = os.path.join(PROJECT_DIR, "dataset", "news_dataset.csv")
        self.assertTrue(os.path.exists(csv_path), "Dataset CSV must exist.")

        df = pd.read_csv(csv_path)
        self.assertIn("title", df.columns)
        self.assertIn("text", df.columns)
        self.assertIn("label", df.columns)
        self.assertGreater(len(df), 50, "Dataset must contain sufficient samples.")

        labels = set(df["label"].unique())
        self.assertTrue(0 in labels and 1 in labels, "Dataset must contain both 0 (Reliable) and 1 (Fake).")

    def test_04_model_artifacts_loaded(self):
        """Test that saved models, vectorizer, and metrics are ready."""
        self.assertTrue(is_model_trained(), "Models must be trained and saved.")
        model, vec, metrics = load_artifacts()
        self.assertIsNotNone(model)
        self.assertIsNotNone(vec)
        self.assertIn("test_metrics", metrics)
        self.assertGreaterEqual(metrics["test_metrics"]["accuracy"], 0.85)

    def test_05_predict_reliable_sample(self):
        """Test prediction on genuine scientific news."""
        title = "James Webb Space Telescope Observes Distant Galaxies"
        body = "NASA astronomers published peer-reviewed findings in Nature detailing distant starlight emitted 350 million years after the Big Bang."
        res = predict_news(title, body)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["prediction_label"], "Likely Reliable")
        self.assertEqual(res["prediction_int"], 0)
        self.assertGreaterEqual(res["confidence"], 50.0)
        self.assertIn("key_tokens", res)

    def test_06_predict_fake_sample(self):
        """Test prediction on fabricated conspiracy news."""
        title = "SECRET LEAK: Aliens Disguised As Politicians Control Government"
        body = "Patriotic whistleblowers leaked documents proving deep state aliens are spraying secret nanodust to brainwash voters. Share this truth before censors delete it!"
        res = predict_news(title, body)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["prediction_label"], "Potentially Fake")
        self.assertEqual(res["prediction_int"], 1)
        self.assertGreaterEqual(res["confidence"], 50.0)

    def test_07_history_manager(self):
        """Test history logging, reading, and clearing."""
        clear_history()
        stats_before = get_history_stats()
        self.assertEqual(stats_before["total_queries"], 0)

        # Log a test prediction
        logged = log_prediction(
            headline="Test Headline",
            article="Test article content for unit verification.",
            prediction="Likely Reliable",
            confidence=95.2,
            model_used="Passive-Aggressive Classifier",
            word_count=7
        )
        self.assertTrue(logged)

        df_hist = load_history()
        self.assertEqual(len(df_hist), 1)
        self.assertEqual(df_hist.iloc[0]["Prediction"], "Likely Reliable")

        stats_after = get_history_stats()
        self.assertEqual(stats_after["total_queries"], 1)
        self.assertEqual(stats_after["reliable_count"], 1)

    def test_08_data_analysis_modules(self):
        """Test data exploration and plot generation in headless mode."""
        csv_path = os.path.join(PROJECT_DIR, "dataset", "news_dataset.csv")
        df = load_dataset(csv_path)

        summary = get_dataset_summary(df)
        self.assertEqual(summary["total_rows"], len(df))

        # Test plot generation (must not raise exceptions)
        fig1 = plot_label_distribution(df)
        self.assertIsNotNone(fig1)

        fig2 = plot_text_length_distribution(df)
        self.assertIsNotNone(fig2)

        # Test n-grams
        ngrams_rel = get_top_ngrams(df, target_label=0, top_n=5)
        self.assertGreater(len(ngrams_rel), 0)

        # Test search
        filtered = search_dataset(df, search_query="NASA", label_filter="All")
        self.assertIsInstance(filtered, pd.DataFrame)


if __name__ == "__main__":
    print("Running Automated Verification Test Suite...")
    unittest.main(verbosity=2)
