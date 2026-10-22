"""
Fake News Detection System - Interactive Web Dashboard
======================================================
Project: Fake News Detection System Using Python and Machine Learning
Target Audience: 3rd-year B.Sc. Artificial Intelligence and Machine Learning
Technology Stack: Streamlit, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Joblib

Pages:
1. Home Dashboard
2. Detect Fake News
3. Dataset Explorer
4. Model Performance
5. Analysis History
6. About the Project
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local modules are accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import LABEL_BADGE_COLORS, decode_label, encode_label
from predict import predict_news, is_model_trained, load_artifacts, METRICS_PATH
from data_analysis import (
    load_dataset,
    get_dataset_summary,
    plot_label_distribution,
    plot_text_length_distribution,
    get_top_ngrams,
    search_dataset
)
from history_manager import log_prediction, load_history, clear_history, get_history_stats

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATASET_PATH = os.path.join(CURRENT_DIR, "dataset", "news_dataset.csv")

# Custom Dark Navy & Blue CSS Styling
CUSTOM_CSS = """
<style>
    /* Global styles */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Metrics Card Container */
    div[data-testid="stMetric"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 0.88rem;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] div {
        color: #38BDF8 !important;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Result Card Styling */
    .result-card {
        border-radius: 12px;
        padding: 24px;
        margin-top: 18px;
        margin-bottom: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .result-reliable {
        background-color: #064E3B;
        border: 2px solid #10B981;
        color: #ECFDF5;
    }
    .result-fake {
        background-color: #7F1D1D;
        border: 2px solid #EF4444;
        color: #FEF2F2;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 24px;
        margin-bottom: 20px;
        color: #CBD5E1;
        font-size: 0.90rem;
        line-height: 1.5;
    }

    /* Workflow card */
    .workflow-step {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 12px;
        height: 100%;
    }

    /* Keyword pills */
    .keyword-pill-fake {
        display: inline-block;
        background: #991B1B;
        color: #FEE2E2;
        border-radius: 14px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #EF4444;
    }
    .keyword-pill-reliable {
        display: inline-block;
        background: #065F46;
        color: #D1FAE5;
        border-radius: 14px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #10B981;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    /* Viva Q&A styling */
    .viva-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Sample Demonstration News Items for 1-Click Examiner Testing
DEMO_SAMPLES = {
    "Select a pre-loaded sample...": ("", ""),
    "Sample 1: Authentic NASA Space Discovery (Reliable)": (
        "James Webb Space Telescope Observes Oldest Galaxies Formed After the Big Bang",
        "Astronomers operating NASA's James Webb Space Telescope have published peer-reviewed findings in the journal Nature detailing the discovery of four of the most distant galaxies ever confirmed. Using high-resolution infrared spectrographs, researchers detected starlight emitted roughly 350 million years after the Big Bang. Astrophysicists stated that these observations provide critical empirical data on how early cosmic structures evolved and seeded modern galaxies."
    ),
    "Sample 2: Authentic Bipartisan Policy News (Reliable)": (
        "Senate Passes Bipartisan Infrastructure Spending Package After Months of Negotiations",
        "The United States Senate voted overwhelmingly to approve a comprehensive 1.2 trillion dollar infrastructure spending package on Tuesday. The bipartisan legislation provides dedicated funding for national highways, bridges, public transit systems, clean drinking water upgrades, and high-speed broadband expansion across rural communities. Congressional budget analysts reported that the bill represents one of the largest federal infrastructure investments in decades."
    ),
    "Sample 3: Fabricated Miracle Cure Hoax (Fake)": (
        "SHOCKING MIRACLE CURE: Boiling Lemon Peels and Baking Soda Destroys 100% of All Cancer Cells",
        "Corrupt pharmaceutical giants and oncology doctors do not want you to know this simple ancient kitchen remedy! Drinking boiled lemon peel water mixed with two tablespoons of pure baking soda instantly alkalizes your bloodstream and starves every tumor within forty-eight hours. Big Pharma has suppressed this natural miracle for decades to preserve their multi-billion dollar chemotherapy monopoly!"
    ),
    "Sample 4: Fabricated Secret Satellite Conspiracy (Fake)": (
        "SECRET LEAK: Shadow Government Operatives Caught Injecting Electronic Ballots via Satellite",
        "Shocking documents exclusively uncovered by underground internet whistleblowers prove that foreign rogue satellites transmitted millions of artificial votes directly into state voting machines during midnight power outages. Anonymous sources inside top-secret agencies claim deep state commanders secretly altered national election outcomes. The mainstream media refuses to broadcast the evidence, but patriotic insiders urge citizens to share this viral truth before it is taken down by global censors!"
    )
}


def load_cached_metrics():
    """
    Safely loads trained evaluation metrics from disk.
    """
    if os.path.exists(METRICS_PATH):
        try:
            import joblib
            return joblib.load(METRICS_PATH)
        except Exception:
            return None
    return None


# Session State Management
if "page" not in st.session_state:
    st.session_state["page"] = "Home Dashboard"
if "headline_input" not in st.session_state:
    st.session_state["headline_input"] = ""
if "article_input" not in st.session_state:
    st.session_state["article_input"] = ""


# Sidebar Navigation & System Status
st.sidebar.image("https://img.icons8.com/fluency/96/news.png", width=64)
st.sidebar.title("Navigation")

PAGES = [
    "Home Dashboard",
    "Detect Fake News",
    "Dataset Explorer",
    "Model Performance",
    "Analysis History",
    "About the Project"
]

selected_page = st.sidebar.radio(
    "Go to page:",
    PAGES,
    index=PAGES.index(st.session_state["page"]) if st.session_state["page"] in PAGES else 0
)
st.session_state["page"] = selected_page

# Sidebar System Health Status
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

dataset_exists = os.path.exists(DATASET_PATH)
models_ready = is_model_trained()

if dataset_exists:
    st.sidebar.markdown("🟢 **Dataset:** `news_dataset.csv` loaded")
else:
    st.sidebar.markdown("🔴 **Dataset:** Missing")

if models_ready:
    metrics = load_cached_metrics()
    best_name = metrics.get("best_model_name", "Passive-Aggressive") if metrics else "Trained"
    st.sidebar.markdown(f"🟢 **Model:** {best_name}")
    if metrics and "test_metrics" in metrics:
        acc = metrics["test_metrics"].get("accuracy", 0.0) * 100
        st.sidebar.markdown(f"🎯 **Test Accuracy:** `{acc:.1f}%`")
else:
    st.sidebar.markdown("🟡 **Model:** Not trained yet")

st.sidebar.markdown("---")
st.sidebar.caption("Fake News Detection System  \nB.Sc. Artificial Intelligence & Machine Learning")


# ==========================================
# PAGE 1: HOME DASHBOARD
# ==========================================
if st.session_state["page"] == "Home Dashboard":
    st.title("📰 Fake News Detection System")
    st.markdown(
        "A machine learning application built with **Python**, **Pandas**, **Scikit-learn**, "
        "and **Streamlit** to detect linguistic markers of misinformation and assist in news credibility evaluation."
    )

    metrics = load_cached_metrics()
    dataset_df = None
    if dataset_exists:
        try:
            dataset_df = load_dataset(DATASET_PATH)
        except Exception:
            dataset_df = None

    # Top Metric Tiles
    st.markdown("### 📊 Project Overview & Real Evaluation Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)

    if metrics and "dataset_stats" in metrics and "test_metrics" in metrics:
        stats = metrics["dataset_stats"]
        test_m = metrics["test_metrics"]
        c1.metric("Total Records", f"{stats.get('total_records', 0):,}")
        c2.metric("Training Samples", f"{stats.get('train_records', 0):,} (80%)")
        c3.metric("Test Samples", f"{stats.get('test_records', 0):,} (20%)")
        c4.metric("Test Accuracy", f"{test_m.get('accuracy', 0.0)*100:.1f}%")
        c5.metric("Test F1-Score", f"{test_m.get('f1_score', 0.0):.4f}")
    elif dataset_df is not None:
        total_rows = len(dataset_df)
        c1.metric("Total Records", f"{total_rows:,}")
        c2.metric("Training Set", f"{int(total_rows*0.8):,} (80%)")
        c3.metric("Testing Set", f"{int(total_rows*0.2):,} (20%)")
        c4.metric("Test Accuracy", "Train Model")
        c5.metric("Test F1-Score", "Train Model")
    else:
        c1.metric("Total Records", "N/A")
        c2.metric("Training Set", "N/A")
        c3.metric("Testing Set", "N/A")
        c4.metric("Test Accuracy", "N/A")
        c5.metric("Test F1-Score", "N/A")

    st.markdown("---")

    # Quick Navigation Call to Action
    col_cta1, col_cta2 = st.columns([3, 1])
    with col_cta1:
        st.info("💡 **Ready to test an article?** Head over to the News Detection page to paste any article or test our pre-loaded samples.")
    with col_cta2:
        if st.button("🚀 Detect Fake News Now", use_container_width=True, type="primary"):
            st.session_state["page"] = "Detect Fake News"
            st.rerun()

    st.markdown("### ⚙️ Machine Learning Pipeline Workflow")
    st.markdown("The project adheres to rigorous data science principles, avoiding lookahead bias and data leakage:")

    w1, w2, w3, w4 = st.columns(4)

    with w1:
        st.markdown(
            """
            <div class="workflow-step">
                <h4>1. Data Preprocessing</h4>
                <p style="color: #94A3B8; font-size: 0.88rem;">
                    Loads labeled news CSV via Pandas. Deduplicates records, cleans noise (URLs, HTML, symbols), and combines headlines with article bodies.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with w2:
        st.markdown(
            """
            <div class="workflow-step">
                <h4>2. TF-IDF Vectorization</h4>
                <p style="color: #94A3B8; font-size: 0.88rem;">
                    Converts text into numerical feature vectors using Term Frequency-Inverse Document Frequency (unigrams + bigrams), fit strictly on training data.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with w3:
        st.markdown(
            """
            <div class="workflow-step">
                <h4>3. Model Training</h4>
                <p style="color: #94A3B8; font-size: 0.88rem;">
                    Trains Passive-Aggressive Classifier and Logistic Regression with stratified split (80/20) and evaluates on the held-out test set.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with w4:
        st.markdown(
            """
            <div class="workflow-step">
                <h4>4. Inference & Explainability</h4>
                <p style="color: #94A3B8; font-size: 0.88rem;">
                    Generates predictions ('Potentially Fake' vs 'Likely Reliable'), computes confidence scores, and extracts key tokens driving the decision.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🎓 Academic Note for B.Sc. AI & ML Evaluators")
    st.write(
        "This project showcases essential data science and machine learning concepts learned in "
        "*Python for Data Science*, including data cleaning with regular expressions, Pandas exploration, "
        "Scikit-learn pipeline construction, TF-IDF feature extraction, model evaluation via Confusion Matrices, "
        "and reproducible model deployment."
    )


# ==========================================
# PAGE 2: DETECT FAKE NEWS
# ==========================================
elif st.session_state["page"] == "Detect Fake News":
    st.title("🔍 Fake News Detection")
    st.markdown("Analyze news headlines and complete articles using the trained machine learning pipeline.")

    if not is_model_trained():
        st.warning("⚠️ **Model Not Trained:** The model files were not found. Please run `python train_model.py` in your terminal to train and serialize the model.")

    # Demonstration Samples Selector
    st.markdown("#### 🧪 Quick Demonstration Samples")
    selected_sample = st.selectbox(
        "Choose an example to populate the input fields instantly:",
        list(DEMO_SAMPLES.keys()),
        index=0
    )

    if selected_sample != "Select a pre-loaded sample...":
        demo_title, demo_text = DEMO_SAMPLES[selected_sample]
        st.session_state["headline_input"] = demo_title
        st.session_state["article_input"] = demo_text

    # Input Form
    with st.form("news_detection_form"):
        headline = st.text_input(
            "News Headline (Optional but Recommended):",
            value=st.session_state["headline_input"],
            placeholder="e.g. NASA Announces Breakthrough Discovery in Space Exploration",
            help="Headlines often contain strong topical and sentiment indicators."
        )

        article = st.text_area(
            "Article Text / Content:",
            value=st.session_state["article_input"],
            height=200,
            placeholder="Paste the full news article body here...",
            help="Provide at least 3 words for accurate feature representation."
        )

        col_b1, col_b2, col_b3 = st.columns([2, 1, 3])
        with col_b1:
            analyze_submitted = st.form_submit_button("🔍 Analyze News", type="primary", use_container_width=True)
        with col_b2:
            clear_submitted = st.form_submit_button("🧹 Clear Input", use_container_width=True)

    if clear_submitted:
        st.session_state["headline_input"] = ""
        st.session_state["article_input"] = ""
        st.rerun()

    # Prediction Execution
    if analyze_submitted:
        with st.spinner("Processing text and running machine learning inference..."):
            result = predict_news(headline=headline, article=article)

        if result.get("status") == "error":
            st.error(f"❌ {result.get('message')}")
        else:
            pred_label = result["prediction_label"]
            confidence = result["confidence"]
            model_name = result["model_name"]
            key_tokens = result["key_tokens"]
            explanation = result["explanation"]
            word_count = result["word_count"]

            # Log to History CSV
            log_prediction(
                headline=headline,
                article=article,
                prediction=pred_label,
                confidence=confidence,
                model_used=model_name,
                word_count=word_count
            )

            # Result Presentation Card
            badge_class = "result-fake" if pred_label == "Potentially Fake" else "result-reliable"
            badge_icon = "⚠️" if pred_label == "Potentially Fake" else "✅"

            st.markdown(
                f"""
                <div class="result-card {badge_class}">
                    <h2 style="margin: 0; color: inherit !important;">{badge_icon} Prediction: {pred_label}</h2>
                    <p style="margin-top: 8px; font-size: 1.05rem; opacity: 0.95;">
                        Classified by <strong>{model_name}</strong> with a model confidence score of <strong>{confidence}%</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Confidence Progress Bar
            st.markdown(f"**Model Confidence Level: {confidence}%**")
            st.progress(min(1.0, confidence / 100.0))

            # Explainability & Contributing Keywords
            st.markdown("#### 🧠 Transparent Model Explainability (Top Informative Features)")
            st.markdown(
                "These are the most significant words identified in the submitted text based on "
                "the model's learned TF-IDF feature weights:"
            )

            if key_tokens:
                pills_html = ""
                for tok in key_tokens:
                    token_str = tok["token"]
                    lean = tok["lean"]
                    pill_class = "keyword-pill-fake" if lean == "Fake" else "keyword-pill-reliable"
                    pill_prefix = "⚠️" if lean == "Fake" else "✅"
                    pills_html += f'<span class="{pill_class}">{pill_prefix} {token_str} ({lean})</span>'
                st.markdown(pills_html, unsafe_allow_html=True)
            else:
                st.info("The submitted terms did not match specific high-weight vocabulary tokens in the model index.")

            st.markdown("#### 📝 Diagnostic Interpretation")
            st.write(explanation)

    # Mandatory Responsible AI Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚖️ Responsible AI Notice:</strong><br>
            This system uses machine learning to identify patterns in a labeled news dataset. 
            Its prediction does not prove whether a news article is true or false. 
            The model may make mistakes, and important claims should be checked against multiple trustworthy sources.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# PAGE 3: DATASET EXPLORER
# ==========================================
elif st.session_state["page"] == "Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.markdown("Inspect, filter, and analyze the training corpus using **Pandas**, **Matplotlib**, and **Seaborn**.")

    if not dataset_exists:
        st.error(f"❌ Dataset not found at: `{DATASET_PATH}`")
        st.info("You can generate the starter dataset by running `python dataset/create_dataset.py` in your terminal.")
    else:
        df = load_dataset(DATASET_PATH)
        summary = get_dataset_summary(df)

        # Overview Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", f"{summary['total_rows']:,}")
        m2.metric("Total Columns", summary["total_cols"])
        m3.metric("Reliable Articles", f"{summary['reliable_count']} ({summary['reliable_pct']}%)")
        m4.metric("Fake Articles", f"{summary['fake_count']} ({summary['fake_pct']}%)")

        st.markdown("---")

        # Dataset Preview
        st.subheader("1. Dataset Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")

        # Structural Inspection & Data Hygiene
        st.subheader("2. Dataset Schema & Missing Value Audit")
        col_sch1, col_sch2 = st.columns([3, 2])

        with col_sch1:
            st.markdown("**Column Information & Null Statistics:**")
            st.dataframe(summary["dtypes_df"], use_container_width=True)

        with col_sch2:
            st.markdown("**Data Integrity Checks:**")
            dup_count = summary["duplicate_count"]
            if dup_count == 0:
                st.success("✅ **Duplicates:** 0 duplicate text records detected. Dataset is clean.")
            else:
                st.warning(f"⚠️ **Duplicates:** {dup_count} duplicate records detected.")

            null_text = df["text"].isnull().sum() if "text" in df.columns else 0
            if null_text == 0:
                st.success("✅ **Article Content:** 0 missing values in 'text' column.")
            else:
                st.warning(f"⚠️ **Article Content:** {null_text} missing values in 'text' column.")

        st.markdown("---")

        # Visualizations
        st.subheader("3. Exploratory Data Visualizations")
        viz1, viz2 = st.columns(2)

        with viz1:
            st.markdown("**Class Balance (Ground Truth):**")
            fig_label = plot_label_distribution(df)
            st.pyplot(fig_label)

        with viz2:
            st.markdown("**Article Word Count Distribution:**")
            fig_len = plot_text_length_distribution(df)
            st.pyplot(fig_len)

        st.markdown("---")

        # Top Keywords by Class
        st.subheader("4. Characteristic N-Grams by Class")
        c_kw1, c_kw2 = st.columns(2)

        with c_kw1:
            st.markdown("🟢 **Top Terms in Likely Reliable Articles:**")
            top_reliable = get_top_ngrams(df, target_label=0, top_n=10)
            st.dataframe(top_reliable, use_container_width=True)

        with c_kw2:
            st.markdown("🔴 **Top Terms in Potentially Fake Articles:**")
            top_fake = get_top_ngrams(df, target_label=1, top_n=10)
            st.dataframe(top_fake, use_container_width=True)

        st.markdown("---")

        # Interactive Search & Filter
        st.subheader("5. Interactive Keyword Search & Filtering")
        f1, f2 = st.columns([3, 1])
        with f1:
            search_query = st.text_input("Search articles by keyword or phrase:", placeholder="e.g. vaccine, NASA, senate...")
        with f2:
            label_filter = st.selectbox("Filter by Class:", ["All", "Likely Reliable (0)", "Potentially Fake (1)"])

        filtered_df = search_dataset(df, search_query=search_query, label_filter=label_filter)
        st.write(f"Showing **{len(filtered_df)}** matching records:")
        st.dataframe(filtered_df, use_container_width=True)


# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================
elif st.session_state["page"] == "Model Performance":
    st.title("📈 Model Evaluation & Performance Metrics")
    st.markdown("Actual evaluation results calculated on the **held-out test set (20%)** during model training.")

    metrics = load_cached_metrics()

    if not metrics or "test_metrics" not in metrics:
        st.warning("⚠️ **Model Metrics Not Found:** Please run `python train_model.py` to train models and generate evaluation metrics.")
    else:
        test_m = metrics["test_metrics"]
        best_name = metrics.get("best_model_name", "Passive-Aggressive Classifier")
        comp_df = metrics.get("comparison_df", pd.DataFrame())

        # Top Metrics
        st.subheader(f"1. Best Model Evaluation: {best_name}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{test_m.get('accuracy', 0.0):.4f}")
        m2.metric("Precision", f"{test_m.get('precision', 0.0):.4f}")
        m3.metric("Recall", f"{test_m.get('recall', 0.0):.4f}")
        m4.metric("F1-Score", f"{test_m.get('f1_score', 0.0):.4f}")

        st.markdown("---")

        # Model Comparison Table & Chart
        st.subheader("2. Model Comparison: Logistic Regression vs. Passive-Aggressive")
        c_tbl, c_chart = st.columns([1, 1])

        with c_tbl:
            st.markdown("**Held-Out Test Set Metrics Comparison:**")
            if not comp_df.empty:
                st.dataframe(comp_df, use_container_width=True)

        with c_chart:
            if not comp_df.empty:
                # Comparison bar chart
                fig_comp, ax_comp = plt.subplots(figsize=(6, 3.5), facecolor="#1E293B")
                ax_comp.set_facecolor("#1E293B")

                metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1_Score"]
                x = np.arange(len(metrics_to_plot))
                width = 0.35

                row_lr = comp_df[comp_df["Model"] == "Logistic Regression"]
                row_pac = comp_df[comp_df["Model"] == "Passive-Aggressive Classifier"]

                vals_lr = [row_lr[m].values[0] for m in metrics_to_plot] if not row_lr.empty else [0]*4
                vals_pac = [row_pac[m].values[0] for m in metrics_to_plot] if not row_pac.empty else [0]*4

                ax_comp.bar(x - width/2, vals_lr, width, label="Logistic Reg.", color="#2563EB", edgecolor="#475569")
                ax_comp.bar(x + width/2, vals_pac, width, label="Passive-Aggr.", color="#38BDF8", edgecolor="#475569")

                ax_comp.set_xticks(x)
                ax_comp.set_xticklabels(metrics_to_plot, color="#F8FAFC", fontsize=9)
                ax_comp.set_ylim(0.7, 1.05)
                ax_comp.set_ylabel("Score", color="#F8FAFC", fontsize=9)
                ax_comp.set_title("Performance Comparison", color="#F8FAFC", fontsize=11, fontweight="bold")
                ax_comp.tick_params(colors="#F8FAFC")
                ax_comp.legend(facecolor="#1E293B", edgecolor="#475569", labelcolor="#F8FAFC")
                ax_comp.grid(axis="y", linestyle="--", alpha=0.3, color="#334155")

                for spine in ax_comp.spines.values():
                    spine.set_color("#475569")

                plt.tight_layout()
                st.pyplot(fig_comp)

        st.markdown("---")

        # Confusion Matrix
        st.subheader("3. Confusion Matrix (Held-Out Test Set)")
        cm_data = np.array(test_m.get("confusion_matrix", [[0, 0], [0, 0]]))

        cm_col1, cm_col2 = st.columns([1, 1])

        with cm_col1:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4), facecolor="#1E293B")
            ax_cm.set_facecolor("#1E293B")

            sns.heatmap(
                cm_data,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Likely Reliable (0)", "Potentially Fake (1)"],
                yticklabels=["Likely Reliable (0)", "Potentially Fake (1)"],
                cbar=False,
                ax=ax_cm,
                annot_kws={"size": 14, "weight": "bold", "color": "#0F172A"}
            )
            ax_cm.set_xlabel("Predicted Label", color="#F8FAFC", fontsize=10, labelpad=8)
            ax_cm.set_ylabel("Actual Ground Truth", color="#F8FAFC", fontsize=10, labelpad=8)
            ax_cm.tick_params(colors="#F8FAFC", labelsize=9)
            ax_cm.set_title(f"Confusion Matrix ({best_name})", color="#F8FAFC", fontsize=11, fontweight="bold", pad=10)

            for spine in ax_cm.spines.values():
                spine.set_color("#475569")

            plt.tight_layout()
            st.pyplot(fig_cm)

        with cm_col2:
            tn, fp, fn, tp = cm_data.ravel() if cm_data.size == 4 else (0, 0, 0, 0)
            st.markdown(
                f"""
                <div class="viva-box">
                    <h4>Confusion Matrix Breakdown:</h4>
                    <ul>
                        <li><strong>True Negatives (TN): {tn}</strong> — Actual reliable news correctly classified as Reliable.</li>
                        <li><strong>False Positives (FP): {fp}</strong> — Reliable news incorrectly flagged as Potentially Fake (Type I Error).</li>
                        <li><strong>False Negatives (FN): {fn}</strong> — Fake news incorrectly classified as Reliable (Type II Error — dangerous in misinformation!).</li>
                        <li><strong>True Positives (TP): {tp}</strong> — Fake news correctly identified as Potentially Fake.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Classification Report Table
        st.subheader("4. Detailed Scikit-Learn Classification Report")
        if "classification_report_dict" in test_m:
            rep_df = pd.DataFrame(test_m["classification_report_dict"]).transpose()
            st.dataframe(rep_df.round(4), use_container_width=True)

        st.markdown("---")

        # Viva Voce Guide Section
        st.subheader("5. 🎓 Viva Voce Guide: Understanding Evaluation Metrics")
        st.markdown(
            """
            External college viva examiners frequently test your understanding of machine learning metrics. 
            Here is how you can explain each metric in simple, articulate English:

            - **Accuracy**: The fraction of total predictions that were correct: `(TP + TN) / Total`. While intuitive, accuracy can be misleading if a dataset has severe class imbalance.
            - **Precision**: `TP / (TP + FP)`. Answers: *"Out of all the news articles the model flagged as Fake, what fraction was actually fake?"* High precision means few false alarms.
            - **Recall (Sensitivity)**: `TP / (TP + FN)`. Answers: *"Out of all actual fake news articles, what fraction did the model successfully catch?"* In fake news detection, high recall is vital to ensure dangerous misinformation does not slip through undetected.
            - **F1-Score**: The harmonic mean of Precision and Recall: `2 * (Precision * Recall) / (Precision + Recall)`. It provides a balanced score when you need both low false alarms and high coverage.
            - **Data Leakage Prevention**: We fit `TfidfVectorizer` **strictly** on the training split (`X_train`) and only call `transform()` on `X_test`. If you fit on the whole dataset before splitting, the model 'cheats' by learning test set vocabulary, producing falsely inflated test metrics.
            """
        )


# ==========================================
# PAGE 5: ANALYSIS HISTORY
# ==========================================
elif st.session_state["page"] == "Analysis History":
    st.title("🕒 Analysis History")
    st.markdown("Review records of news headlines and articles analyzed in previous sessions.")

    history_df = load_history()
    h_stats = get_history_stats()

    # Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Analyses", h_stats["total_queries"])
    c2.metric("Reliable Detections", h_stats["reliable_count"])
    c3.metric("Fake Detections", h_stats["fake_count"])
    c4.metric("Avg. Confidence", f"{h_stats['avg_confidence']}%")

    st.markdown("---")

    if history_df.empty:
        st.info("ℹ️ No news articles have been analyzed yet. Head over to **Detect Fake News** to submit an article!")
    else:
        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            history_filter = st.selectbox("Filter History by Prediction:", ["All", "Likely Reliable", "Potentially Fake"])
        with col_b2:
            if st.button("🗑️ Clear Analysis History", type="secondary"):
                clear_history()
                st.success("Analysis history cleared successfully.")
                st.rerun()

        if history_filter != "All":
            history_df = history_df[history_df["Prediction"] == history_filter]

        st.dataframe(history_df, use_container_width=True)

        st.caption("Data is stored locally in `history/analysis_history.csv` without external server tracking.")


# ==========================================
# PAGE 6: ABOUT THE PROJECT
# ==========================================
elif st.session_state["page"] == "About the Project":
    st.title("ℹ️ About the Fake News Detection System")
    st.markdown(
        "A final-year academic project engineered for **B.Sc. Artificial Intelligence and Machine Learning** students."
    )

    st.subheader("1. System Architecture & Tech Stack")
    st.markdown(
        """
        - **Programming Language:** Python 3.10+
        - **Web Framework:** Streamlit (100% Python-based interactive frontend)
        - **Data Manipulation:** Pandas & NumPy
        - **Natural Language Processing:** Scikit-learn `TfidfVectorizer` (unigrams + bigrams, sublinear term frequency scaling)
        - **Classifiers:** 
          - `PassiveAggressiveClassifier` (Online large-margin algorithm suited for streaming text)
          - `LogisticRegression` (Linear probabilistic baseline with L2 regularization)
        - **Visualization:** Matplotlib & Seaborn
        - **Model Persistence:** Joblib
        - **Local Storage:** CSV format
        """
    )

    st.markdown("---")

    st.subheader("2. Academic Project Questions & Answers (Viva Voce Cheat Sheet)")
    st.markdown("Prepare for your college examination and viva defense with these structured answers:")

    viva_qa = [
        ("Q1: What is the main problem statement of this project?",
         "The objective is to apply Natural Language Processing and supervised Machine Learning to classify news headlines and articles into 'Potentially Fake' or 'Likely Reliable' categories based on learned linguistic and lexical patterns from verified training datasets."),

        ("Q2: Why did you use TF-IDF instead of CountVectorizer (Bag of Words)?",
         "CountVectorizer merely counts term occurrences, heavily weighting ubiquitous words. TF-IDF (Term Frequency-Inverse Document Frequency) balances word frequency within an article against how rarely it appears across the entire corpus. This downweights common generic terms while boosting distinctive, highly informative terms that characterize fake vs reliable reporting."),

        ("Q3: Why was Passive-Aggressive Classifier chosen?",
         "The Passive-Aggressive algorithm is a large-margin online learning classifier. It is 'passive' when a training sample is classified correctly with sufficient margin, and 'aggressive' (updating weights) whenever a misclassification occurs. It converges rapidly on sparse, high-dimensional text feature spaces and is widely used for news streams."),

        ("Q4: Why was Logistic Regression also implemented?",
         "Logistic Regression provides an interpretable linear baseline that directly estimates posterior class probabilities via the sigmoid link function. Comparing Passive-Aggressive with Logistic Regression validates model robustness across different algorithmic approaches."),

        ("Q5: What is Data Leakage and how did your pipeline prevent it?",
         "Data leakage occurs when information from outside the training dataset influences model training. We strictly split our dataset into training (80%) and testing (20%) sets before vectorization. The `TfidfVectorizer` was fit only on `X_train`, and only used to transform `X_test`. We also dropped duplicate text rows beforehand to prevent identical samples from appearing in both splits."),

        ("Q6: In fake news detection, which metric is more critical: Precision or Recall?",
         "Both are important, but **Recall for the Fake class** is often critical. A False Negative (letting dangerous fake news pass as reliable) can cause public harm. However, a False Positive damages trustworthy journalism. Hence, the **F1-score** is the primary criterion for model selection."),

        ("Q7: Can a machine learning model independently verify factual truth?",
         "No. The model identifies statistical linguistic correlations and stylistic vocabulary patterns (e.g., sensationalism, clickbait phrasing, conspiracy markers). It does not consult external live knowledge graphs or conduct journalistic investigative reporting. This is why our application explicitly displays a Responsible AI notice."),

        ("Q8: How could this project be improved in future work?",
         "Future enhancements could integrate transformer-based models (such as BERT or RoBERTa), automated web fact-checking API integration (e.g., Google Fact Check Tools API), multilingual news support, and multimodal analysis (verifying images alongside text).")
    ]

    for q, a in viva_qa:
        st.markdown(
            f"""
            <div class="viva-box">
                <h4 style="color: #38BDF8 !important; margin-bottom: 6px;">{q}</h4>
                <p style="color: #CBD5E1; margin: 0; line-height: 1.6;">{a}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("3. Licensing & Responsible Use")
    st.write(
        "This project is developed solely for educational and academic research purposes under the open-source MIT License. "
        "No paid APIs, proprietary software, or commercial AI services are required."
    )
