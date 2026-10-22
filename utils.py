"""
Text Preprocessing and Utility Functions
=========================================
Project: Fake News Detection System Using Python and Machine Learning
Target Audience: 3rd-year B.Sc. Artificial Intelligence and Machine Learning

This module provides reusable preprocessing routines for:
1. Cleaning raw text (lowercasing, removing URLs, HTML tags, special characters).
2. Combining headlines and article bodies into unified text features.
3. Standardizing and encoding/decoding classification labels.
4. Validating user input before feeding it into the ML inference pipeline.
"""

import re
from typing import Tuple, Union


# Standard label mappings used across the entire application.
# 0 -> Likely Reliable (Real news)
# 1 -> Potentially Fake (Misinformation/Disinformation)
LABEL_NAMES = {
    0: "Likely Reliable",
    1: "Potentially Fake"
}

LABEL_BADGE_COLORS = {
    "Likely Reliable": {
        "bg": "#064E3B",      # Deep emerald
        "border": "#10B981",  # Vibrant emerald
        "text": "#D1FAE5",    # Soft mint text
        "icon": "✅"
    },
    "Potentially Fake": {
        "bg": "#7F1D1D",      # Deep crimson
        "border": "#EF4444",  # Vibrant red
        "text": "#FEE2E2",    # Soft rose text
        "icon": "⚠️"
    }
}


def clean_text(text: Union[str, None]) -> str:
    """
    Cleans raw text data by removing noise, markup, and special characters.

    Preprocessing Steps:
    1. Null handling: returns empty string if text is None or not a string.
    2. Lowercasing: ensures vocabulary uniformity (e.g., 'Election' == 'election').
    3. URL removal: strips web links (http, https, www).
    4. HTML tag stripping: removes any remaining web tags.
    5. Special character cleaning: keeps only letters, digits, and basic whitespace.
    6. Whitespace normalization: collapses multiple spaces, tabs, and newlines.

    Parameters
    ----------
    text : str or None
        Raw input string (headline, article, or combined content).

    Returns
    -------
    str
        Cleaned, normalized text ready for TF-IDF feature extraction.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Convert to lowercase
    cleaned = text.lower()

    # Remove URLs (http://, https://, www.)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Remove HTML tags (e.g. <br>, <p>, &amp;)
    cleaned = re.sub(r"<.*?>", " ", cleaned)

    # Remove email addresses
    cleaned = re.sub(r"\S+@\S+", " ", cleaned)

    # Keep only letters, digits, and standard whitespace
    # Non-alphanumeric symbols like @, #, $, %, ^, &, *, ~, etc. are noise in text classification
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", cleaned)

    # Collapse multiple whitespace characters into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def combine_content(title: Union[str, None], text: Union[str, None]) -> str:
    """
    Combines headline (title) and article body into a single composite text.
    Headlines contain strong topical signals, while the body provides contextual depth.

    Parameters
    ----------
    title : str or None
        News article headline.
    text : str or None
        News article body text.

    Returns
    -------
    str
        Combined string.
    """
    title_str = str(title).strip() if title is not None else ""
    text_str = str(text).strip() if text is not None else ""

    if title_str and text_str:
        return f"{title_str}. {text_str}"
    elif title_str:
        return title_str
    elif text_str:
        return text_str
    return ""


def encode_label(raw_label: Union[str, int, float]) -> int:
    """
    Standardizes various dataset target labels into binary integer format:
    0 = Likely Reliable / Real
    1 = Potentially Fake / Fake

    Supports formats like:
    - 0, 1
    - 'REAL', 'FAKE'
    - 'reliable', 'fake'
    - 'true', 'false'

    Parameters
    ----------
    raw_label : str, int, or float
        Raw label value from dataset.

    Returns
    -------
    int
        0 for Reliable, 1 for Fake.
    """
    if isinstance(raw_label, (int, float)):
        # Already binary integer or float
        return 1 if int(raw_label) == 1 else 0

    label_str = str(raw_label).strip().lower()

    if label_str in {"fake", "1", "false", "unreliable", "potentially fake", "spam"}:
        return 1
    elif label_str in {"real", "0", "true", "reliable", "likely reliable"}:
        return 0
    else:
        # Default fallback
        return 0


def decode_label(label_int: int) -> str:
    """
    Maps binary integer prediction back to human-readable label string.
    """
    return LABEL_NAMES.get(int(label_int), "Likely Reliable")


def validate_input_text(headline: str, article: str) -> Tuple[bool, str]:
    """
    Validates user input submitted in the detection form.

    Checks:
    1. At least one field (headline or article) contains non-whitespace text.
    2. Total word count is at least 3 words to allow meaningful feature extraction.

    Parameters
    ----------
    headline : str
        User-provided headline.
    article : str
        User-provided article body.

    Returns
    -------
    tuple of (is_valid: bool, error_message: str)
    """
    combined = combine_content(headline, article).strip()

    if not combined:
        return False, "Please enter a news headline or paste an article text to analyze."

    word_count = len(combined.split())
    if word_count < 3:
        return False, f"The input is too short ({word_count} words). Please provide at least 3 words for accurate analysis."

    return True, ""
