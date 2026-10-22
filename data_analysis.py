"""
Pandas Data Exploration and Visualization Utilities
===================================================
Project: Fake News Detection System Using Python and Machine Learning
Target Audience: 3rd-year B.Sc. Artificial Intelligence and Machine Learning

This module provides data analysis and visualization functions:
1. Computes dataset structural summaries (shapes, dtypes, nulls, duplicates).
2. Generates Seaborn/Matplotlib charts for label balance and text length distributions.
3. Performs n-gram frequency extraction to compare vocabulary across classes.
4. Offers keyword search and filtering for interactive Streamlit exploration.
"""

import os
import sys
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless web execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import clean_text, decode_label, encode_label

# Styling constants matching dark navy & blue theme
PALETTE_COLORS = {
    "Reliable": "#2563EB",  # Blue
    "Fake": "#EF4444",      # Red
    "Background": "#0F172A",
    "Card": "#1E293B",
    "Text": "#F8FAFC",
    "Grid": "#334155"
}


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Safely loads the dataset from a CSV file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")
    return pd.read_csv(csv_path)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Calculates detailed summary metrics for dataset exploration.

    Returns
    -------
    dict
        Dictionary containing shapes, dtypes, missing values, duplicates, and class balance.
    """
    total_rows, total_cols = df.shape

    # Data types table
    dtypes_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Null Percentage (%)": ((df.isnull().sum().values / total_rows) * 100).round(2)
    })

    # Duplicates analysis
    # Check duplicates based on text if available, or entire row
    if "text" in df.columns:
        duplicate_count = int(df.duplicated(subset=["text"]).sum())
    else:
        duplicate_count = int(df.duplicated().sum())

    # Label distribution
    if "label" in df.columns:
        # Standardize labels
        standardized_labels = df["label"].apply(encode_label)
        label_counts = standardized_labels.value_counts().to_dict()
        reliable_count = label_counts.get(0, 0)
        fake_count = label_counts.get(1, 0)
        reliable_pct = round((reliable_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        fake_pct = round((fake_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
    else:
        reliable_count, fake_count, reliable_pct, fake_pct = 0, 0, 0.0, 0.0

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "columns": list(df.columns),
        "dtypes_df": dtypes_df,
        "duplicate_count": duplicate_count,
        "reliable_count": reliable_count,
        "fake_count": fake_count,
        "reliable_pct": reliable_pct,
        "fake_pct": fake_pct
    }


def plot_label_distribution(df: pd.DataFrame):
    """
    Generates a Seaborn bar chart showing the balance of Fake vs Reliable articles.
    """
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=PALETTE_COLORS["Card"])
    ax.set_facecolor(PALETTE_COLORS["Card"])

    standardized_labels = df["label"].apply(encode_label)
    counts = standardized_labels.value_counts().sort_index()

    labels = ["Likely Reliable (0)", "Potentially Fake (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = [PALETTE_COLORS["Reliable"], PALETTE_COLORS["Fake"]]

    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="#475569", linewidth=1.2)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height} ({height/len(df)*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            color=PALETTE_COLORS["Text"],
            fontsize=10, fontweight="bold"
        )

    ax.set_title("Dataset Class Balance (Ground Truth)", color=PALETTE_COLORS["Text"], fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Article Count", color=PALETTE_COLORS["Text"], fontsize=10)
    ax.tick_params(colors=PALETTE_COLORS["Text"], labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3, color=PALETTE_COLORS["Grid"])

    for spine in ax.spines.values():
        spine.set_color("#475569")

    plt.tight_layout()
    return fig


def plot_text_length_distribution(df: pd.DataFrame):
    """
    Generates a KDE distribution plot of article word counts separated by class.
    """
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=PALETTE_COLORS["Card"])
    ax.set_facecolor(PALETTE_COLORS["Card"])

    df_copy = df.copy()
    if "text" in df_copy.columns:
        df_copy["word_count"] = df_copy["text"].fillna("").apply(lambda x: len(str(x).split()))
        df_copy["standard_label"] = df_copy["label"].apply(encode_label).map({0: "Likely Reliable", 1: "Potentially Fake"})

        sns.histplot(
            data=df_copy,
            x="word_count",
            hue="standard_label",
            kde=True,
            palette={"Likely Reliable": PALETTE_COLORS["Reliable"], "Potentially Fake": PALETTE_COLORS["Fake"]},
            ax=ax,
            bins=20,
            alpha=0.4,
            edgecolor="#475569"
        )

        ax.set_title("Article Word Count Distribution by Class", color=PALETTE_COLORS["Text"], fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Word Count per Article", color=PALETTE_COLORS["Text"], fontsize=10)
        ax.set_ylabel("Density / Frequency", color=PALETTE_COLORS["Text"], fontsize=10)
        ax.tick_params(colors=PALETTE_COLORS["Text"], labelsize=10)
        ax.grid(axis="both", linestyle="--", alpha=0.3, color=PALETTE_COLORS["Grid"])

        # Style legend
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(PALETTE_COLORS["Card"])
            legend.get_frame().set_edgecolor("#475569")
            for text in legend.get_texts():
                text.set_color(PALETTE_COLORS["Text"])

        for spine in ax.spines.values():
            spine.set_color("#475569")

    plt.tight_layout()
    return fig


def get_top_ngrams(df: pd.DataFrame, target_label: int, top_n=10) -> pd.DataFrame:
    """
    Extracts the most frequent unigrams and bigrams for a given class.
    """
    df_copy = df.copy()
    df_copy["clean"] = (df_copy["title"].fillna("") + " " + df_copy["text"].fillna("")).apply(clean_text)
    subset = df_copy[df_copy["label"].apply(encode_label) == target_label]["clean"]

    if subset.empty:
        return pd.DataFrame(columns=["Token", "Frequency"])

    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=top_n)
    bow = vec.fit_transform(subset)
    counts = np.array(bow.sum(axis=0)).flatten()
    tokens = vec.get_feature_names_out()

    result_df = pd.DataFrame({"Token": tokens, "Frequency": counts}).sort_values(by="Frequency", ascending=False).reset_index(drop=True)
    return result_df


def search_dataset(df: pd.DataFrame, search_query: str = "", label_filter: str = "All") -> pd.DataFrame:
    """
    Filters the dataset by search term and label category.
    """
    filtered = df.copy()

    # Apply label filter
    if label_filter == "Likely Reliable (0)":
        filtered = filtered[filtered["label"].apply(encode_label) == 0]
    elif label_filter == "Potentially Fake (1)":
        filtered = filtered[filtered["label"].apply(encode_label) == 1]

    # Apply search query
    if search_query.strip():
        q = search_query.strip().lower()
        title_match = filtered["title"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
        text_match = filtered["text"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
        filtered = filtered[title_match | text_match]

    return filtered
