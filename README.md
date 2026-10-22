# Fake News Detection System Using Python and Machine Learning

An interactive, Python-centric machine learning application engineered for third-year **B.Sc. Artificial Intelligence and Machine Learning** students to demonstrate data analysis, feature engineering, text classification, model evaluation, and web deployment.

---

## 📌 1. Project Overview

The **Fake News Detection System** accepts news headlines and article text and predicts whether the content resembles examples labeled as **"Potentially Fake"** or **"Likely Reliable"** based on patterns learned from a labeled training dataset.

### Key Objectives:
- **Zero Frontend Fragmentation:** 100% Python-centric architecture using **Streamlit** (no JavaScript/HTML/React needed).
- **Dual Model Comparative Architecture:** Implements and compares both **Passive-Aggressive Classifier** (large-margin online classifier) and **Logistic Regression** (probabilistic baseline).
- **Explainable AI (XAI):** Features an interpretability engine that identifies the specific keywords/n-grams that drove the model's classification.
- **Responsible AI Disclaimer:** Prominently clarifies that the model evaluates stylistic and lexical patterns in text rather than asserting omniscient factual truth.
- **Academic Rigor:** Built specifically for college presentations, project submissions, and viva voce defense.

---

## 🛠️ 2. Technology Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Core Language** | Python 3.10+ | Primary programming language |
| **Interactive Dashboard** | Streamlit | Modern dark navy & blue dashboard UI |
| **Data Manipulation** | Pandas & NumPy | Dataset ingestion, deduplication, cleaning, array math |
| **Machine Learning** | Scikit-learn | TF-IDF vectorization, Logistic Regression, Passive-Aggressive |
| **Feature Extraction** | `TfidfVectorizer` | Sublinear TF-IDF with unigrams & bigrams (max 5,000 features) |
| **Data Visualization** | Matplotlib & Seaborn | Class balance, word-length histograms, confusion matrix heatmaps |
| **Model Persistence** | Joblib | Serialization of pipelines, fitted models, and test metrics |
| **Data Storage** | CSV | Dataset storage and user analysis history tracking |

---

## 📂 3. Project Directory Structure

```
Fake-News-Detection-System/
├── app.py                      # Multi-page Streamlit application (6 interactive pages)
├── train_model.py              # ML training pipeline (data cleaning, TF-IDF, training, evaluation, saving)
├── data_analysis.py            # Pandas analysis functions, summary statistics, chart generators
├── predict.py                  # Real-time inference engine, confidence scoring, keyword explainability
├── utils.py                    # Text preprocessing, normalization, and label encoding helpers
├── history_manager.py          # Persistent query history logger (CSV storage, retrieve, clear)
├── test_system.py              # Automated unit and integration test suite
├── requirements.txt            # Python dependencies specification
├── README.md                   # Complete academic documentation and viva cheat sheet
├── .streamlit/
│   └── config.toml             # Streamlit theme (Dark navy, blue, white) & headless settings
├── dataset/
│   ├── news_dataset.csv        # Curated, balanced starter dataset (Reliable vs Fake news across domains)
│   ├── create_dataset.py       # Script to build or regenerate starter dataset
│   └── README.md               # Dataset documentation, column schema, benchmark sources (WELFake/ISOT)
├── models/
│   ├── news_model.joblib       # Best trained classification model
│   ├── tfidf_vectorizer.joblib # Fitted TF-IDF vectorizer
│   ├── news_pipeline.joblib    # Unified Scikit-Learn inference pipeline
│   ├── logistic_regression.joblib # Trained Logistic Regression model
│   ├── passive_aggressive.joblib  # Trained Passive-Aggressive model
│   └── model_metrics.joblib    # Actual test set evaluation metrics, confusion matrix, comparison data
└── history/
    └── analysis_history.csv    # Query history with timestamp, text preview, prediction, confidence
```

---

## 💻 4. Step-by-Step Windows & VS Code Setup

Follow these instructions to run the project locally on Windows using Visual Studio Code:

### Step 1: Open the Project in VS Code
1. Launch **Visual Studio Code**.
2. Go to **File** > **Open Folder...**
3. Select the folder:
   ```
   C:\Users\ritesh\.gemini\antigravity\scratch\Fake-News-Detection-System
   ```
4. Open the integrated terminal (`Ctrl` + `` ` ``).

### Step 2: Create and Activate a Python Virtual Environment
In the VS Code terminal (PowerShell), run:
```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# If PowerShell restricts scripts, enable script execution:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```
*(You will see `(venv)` appear at the start of your terminal prompt).*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Verify or Generate Dataset
The repository already includes `dataset/news_dataset.csv`. If you ever need to recreate it:
```powershell
python dataset\create_dataset.py
```

### Step 5: Train the Machine Learning Pipeline
Train both classification models on the dataset and compute held-out test evaluation metrics:
```powershell
python train_model.py
```
*Output will display the train/test split, vocabulary size, side-by-side model comparison, and serialization confirmations in `models/`.*

### Step 6: Run Automated Tests (Optional but Recommended)
Verify that all system components pass unit testing:
```powershell
python test_system.py
```

### Step 7: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Your default web browser will automatically open:
```
Local URL: http://localhost:8501
```

---

## 🧠 5. Machine Learning Pipeline Explained

```mermaid
flowchart LR
    A[Raw News CSV] --> B[Text Cleaning & Normalization]
    B --> C[Stratified Split 80/20]
    C --> D[TfidfVectorizer fit on X_train]
    D --> E1[Passive-Aggressive Classifier]
    D --> E2[Logistic Regression]
    E1 --> F[Evaluate on Held-out X_test]
    E2 --> F
    F --> G[Select Best Model & Persist via Joblib]
    G --> H[Streamlit Web Inference]
```

### 1. Data Ingestion & Preprocessing (`utils.py`)
- **Missing Value Handling:** Replaces NaN values in headlines or bodies with empty strings.
- **Deduplication:** Drops duplicate articles before splitting to guarantee no identical texts appear in both training and test sets.
- **Regex Cleaning:** Lowercases text, strips HTML tags (`<.*?>`), removes hyperlinks (`https?://\S+`), and filters non-alphanumeric noise while preserving whitespace.

### 2. Feature Extraction: TF-IDF Vectorization
Text cannot be directly consumed by mathematical algorithms. We transform text into numerical vectors using **TF-IDF**:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
- **Term Frequency (TF):** Measures how frequently token $t$ appears in document $d$.
- **Inverse Document Frequency (IDF):** Penalizes ubiquitous words (e.g. "said", "today") and rewards words unique to specific genres.
- We configure:
  - `ngram_range=(1, 2)`: Captures single words and two-word phrases (e.g., *"secret leak"*, *"peer-reviewed"*).
  - `max_features=5000`: Caps vocabulary size to the 5,000 most informative features.
  - `stop_words='english'`: Filters out common grammatical filler words.

### 3. Model Training & Comparison
We train two complementary linear classification models:
1. **Passive-Aggressive Classifier:**
   - Online learning algorithm.
   - If a training sample is classified correctly with sufficient margin, parameters remain unchanged (**passive**).
   - If misclassified, parameters aggressively update to minimize hinge loss (**aggressive**).
   - Well-suited for high-dimensional, sparse text matrices.
2. **Logistic Regression:**
   - Probabilistic baseline utilizing the sigmoid link function:
     $$P(y=1|x) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
   - Provides smooth, calibrated posterior class probabilities.

### 4. Zero Data Leakage Guarantee
A critical academic point for viva presentations:
> **The vectorizer is fit strictly on `X_train` (`fit_transform`) and only used to transform `X_test` (`transform`).**  
> We never fit the vectorizer on the full dataset before splitting, which would allow vocabulary frequencies from the test set to bleed into training.

---

## 🖥️ 6. Streamlit Web Dashboard Walkthrough

The application features 6 dedicated pages accessible via the sidebar:

1. **🏠 Home Dashboard:**
   - Displays real-time metrics loaded from `model_metrics.joblib`: Total records, Train/Test split sizes, Test Accuracy, and Test F1-Score.
   - Visual step-by-step roadmap explaining the data science lifecycle.
   - Quick CTA button navigating to the detection page.
2. **🔍 Detect Fake News:**
   - Interactive inputs for headline and full article text.
   - **4 Quick Demonstration Samples** allowing 1-click loading of authentic vs fabricated news.
   - Real-time prediction with status cards (`⚠️ Potentially Fake` vs `✅ Likely Reliable`).
   - Confidence percentage meter and diagnostic explanations.
   - **Explainability Pill Badges:** Highlights specific n-grams from the article that pushed the prediction toward Fake or Reliable.
   - Prominent **Responsible AI Disclaimer**.
3. **📊 Dataset Explorer:**
   - Interactive Pandas DataFrame preview (`df.head()`).
   - Schema audit: column types, non-null counts, duplicate row audit.
   - Visualizations: Class balance bar chart and article word count KDE distributions.
   - Most frequent n-grams in Reliable vs Fake classes.
   - Interactive keyword search and class filter.
4. **📈 Model Performance:**
   - Held-out test set metrics: Accuracy, Precision, Recall, and F1-Score.
   - Side-by-side comparison chart: Logistic Regression vs Passive-Aggressive Classifier.
   - Annotated Confusion Matrix heatmap (True Negatives, False Positives, False Negatives, True Positives).
   - Scikit-learn Classification Report table.
   - Dedicated Viva Voce Guide breaking down metric formulas and real-world implications.
5. **🕒 Analysis History:**
   - Automatically logs user predictions to `history/analysis_history.csv`.
   - Displays timestamp, headline, article snippet, prediction, confidence, and word count.
   - Includes summary analytics and a "Clear History" button.
6. **ℹ️ About the Project:**
   - Complete academic documentation, theoretical explanations, ethical limitations, and 12+ viva exam questions and answers.

---

## 🎬 7. Project Demonstration Guide (For Evaluators)

When demonstrating this project to a teacher, examiner, or class:

1. **Start on the Home Dashboard:**
   - Explain the motivation: combating online misinformation using machine learning.
   - Point to the metrics row to show that results are **real** and derived from a held-out test split, not hardcoded.
2. **Demonstrate 1-Click Samples on "Detect Fake News":**
   - Click **Sample 1 (Authentic Space Discovery)**: Show the `✅ Likely Reliable` badge, the confidence score, and the positive key tokens like `perseverance`, `rover`, `organic`.
   - Click **Sample 3 (Miracle Cure Hoax)**: Show the `⚠️ Potentially Fake` badge, the confidence score, and alarmist keywords like `baking soda`, `miracle`, `destroy`.
   - Point out the **Responsible AI Notice** and explain why machine learning detects text style, not factual ground truth.
3. **Showcase the Dataset Explorer:**
   - Show how Pandas handles data inspection, missing value audits, and deduplication.
   - Show the Seaborn class distribution and word length distribution plots.
4. **Present the Model Performance Page:**
   - Explain the Confusion Matrix: discuss the difference between Type I (FP) and Type II (FN) errors in misinformation detection.
   - Explain why the Passive-Aggressive Classifier and Logistic Regression were compared.

---

## ❓ 8. Viva Voce Cheat Sheet (Q&A for Students)

Here are the most common technical questions external examiners ask during third-year B.Sc. AI & ML project vivas:

#### Q1: What is TF-IDF and why use it over CountVectorizer (Bag of Words)?
**Answer:**  
CountVectorizer only computes raw term frequency ($\text{TF}$). In long texts, common grammatical words dominate the vector space without carrying discriminative meaning. **TF-IDF** multiplies term frequency by inverse document frequency ($\text{IDF}$), which heavily penalizes ubiquitous terms across the corpus and boosts distinctive, topical terms that clearly differentiate fake from real articles.

#### Q2: How does the Passive-Aggressive Classifier work?
**Answer:**  
Passive-Aggressive is an online, margin-based algorithm. It remains **passive** when an incoming sample is correctly classified with a margin greater than or equal to 1 (loss is zero). If a prediction is incorrect or violates the margin, it **aggressively** updates the weight vector with the minimal step necessary to satisfy the margin constraint while retaining previously learned knowledge.

#### Q3: Why is Logistic Regression also included?
**Answer:**  
Logistic Regression is the benchmark generalized linear model for binary classification. It models log-odds as a linear combination of features and passes them through a sigmoid function to produce well-calibrated posterior probabilities $P(y=1|x)$. It serves as a transparent comparative baseline to validate the Passive-Aggressive model.

#### Q4: What is Data Leakage and how did you prevent it in your code?
**Answer:**  
Data leakage occurs when test set information inadvertently influences the training process, resulting in unrealistically optimistic validation scores. We prevented this by:
1. Splitting the raw text data into training (80%) and testing (20%) sets using `train_test_split(..., stratify=y)` *before* vectorization.
2. Calling `fit_transform()` on `X_train` only, and only `transform()` on `X_test`.
3. Dropping duplicate articles before splitting to prevent identical records from appearing in both sets.

#### Q5: In fake news detection, is Precision or Recall more important?
**Answer:**  
Both matter, but **Recall for the Fake class** is generally prioritized. A False Negative (labeling a dangerous fake news piece as reliable) allows harmful misinformation to spread unchecked. However, low Precision causes trustworthy news to be falsely flagged, eroding user trust. Hence, we optimize for the **F1-score**, which represents the harmonic mean of Precision and Recall.

#### Q6: What does the Confusion Matrix show?
**Answer:**  
- **True Negatives (TN):** Reliable news correctly predicted as Reliable.
- **False Positives (FP):** Reliable news incorrectly flagged as Potentially Fake (Type I Error).
- **False Negatives (FN):** Fake news incorrectly labeled as Reliable (Type II Error).
- **True Positives (TP):** Fake news correctly identified as Potentially Fake.

#### Q7: Can this system prove whether a news claim is factually true?
**Answer:**  
No. Supervised NLP models detect statistical linguistic patterns, lexical styling, and rhetorical framing characteristic of training examples. They do not query live relational knowledge graphs or verify real-world facts. This is why our system labels items as "Potentially Fake" or "Likely Reliable" and displays an ethical disclaimer.

#### Q8: What are potential future enhancements for this project?
**Answer:**  
1. Upgrading to deep contextual transformers like BERT or RoBERTa.
2. Integrating live fact-checking APIs (e.g. Google Fact Check Tools API).
3. Implementing multimodal fake news detection (verifying images and text together).
4. Expanding cross-lingual detection for multilingual regional news.

---

## 🔧 9. Troubleshooting Guide

| Issue | Cause | Solution |
|:---|:---|:---|
| `Set-ExecutionPolicy` error in PowerShell | Windows restricts unsigned scripts | Run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `FileNotFoundError: news_model.joblib` | Model has not been trained yet | Run: `python train_model.py` before starting Streamlit |
| `DLL load failed while importing _datasets_pair` | Windows Zone.Identifier restriction on `.pyd` files | Run: `Get-ChildItem -Recurse 'venv\Lib\site-packages' \| Unblock-File` |
| Streamlit port 8501 is already in use | Another process is using port 8501 | Run on a different port: `streamlit run app.py --server.port 8502` |

---

## 📜 10. License

This project is released under the **MIT Open Source License**. It is freely available for educational, academic, and non-commercial research use.
