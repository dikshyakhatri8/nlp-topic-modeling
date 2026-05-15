# 📰 NLP Text Classification for Topic Modeling
### Course Project - DLBAIPNLP01

A complete text classification pipeline that assigns newsgroup documents to topics using the **20 Newsgroups** dataset. Three classical ML models are trained, tuned, and evaluated on both a small (4-category) and full (20-category) setup.


## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Visualizations](#visualizations)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Key Findings](#key-findings)


## Project Overview

This project implements a **supervised text classification** system for topic modeling using traditional NLP techniques. The goal is to automatically assign a newsgroup document to one of 20 topic categories based purely on its text content.

**Key techniques used:**
- Text preprocessing (lowercasing, punctuation removal, stop word filtering, lemmatization)
- TF-IDF feature extraction (10,000 features)
- Three classifier models: Naive Bayes, SVM, and Logistic Regression
- 5-fold cross-validation and GridSearch hyperparameter tuning
- Evaluation via accuracy, F1-score, confusion matrix, and per-category analysis


## Dataset

**20 Newsgroups** - a standard NLP benchmark dataset from `sklearn.datasets`.

| Split | Documents |
|-------|-----------|
| Training | 11,314 |
| Test | 7,532 |
| Categories | 20 |

Headers, footers, and quoted replies were removed to prevent data leakage. The dataset spans topics across computers, recreation, science, politics, and religion.


## Pipeline

```
Raw Text
   │
   ▼
Preprocessing
  ├── Lowercase
  ├── Remove numbers & punctuation
  ├── Tokenize
  ├── Remove stop words (NLTK)
  └── Lemmatize (WordNetLemmatizer)
   │
   ▼
TF-IDF Vectorization (max 10,000 features)
   │
   ▼
Model Training & Evaluation
  ├── Naive Bayes (MultinomialNB)
  ├── SVM (LinearSVC)
  └── Logistic Regression
   │
   ▼
Metrics: Accuracy · F1 · Confusion Matrix
```

TF-IDF outperformed raw CountVectorizer by **+1.73%** on Naive Bayes and was selected as the main vectorizer.


## Models

| Model | Notes |
|-------|-------|
| **Naive Bayes** | `MultinomialNB`; tuned `alpha` via GridSearch (best: 0.1) |
| **SVM** | `LinearSVC` with `max_iter=2000` |
| **Logistic Regression** | `C=1.0`, `max_iter=1000`; tuned `C` via GridSearch (best: 5.0) |

5-fold cross-validation was run on the full training set for all three models.


## Results

### Small Dataset (4 categories: hockey, space, graphics, politics.guns)

| Model | Accuracy |
|-------|----------|
| Naive Bayes | **88.94%** |
| SVM | 87.26% |
| Logistic Regression | 88.23% |

### Full Dataset (20 categories)

| Model | Accuracy | CV Mean | CV Std |
|-------|----------|---------|--------|
| Naive Bayes | 66.08% | 70.34% | ±0.57% |
| SVM | 64.98% | 71.96% | ±0.78% |
| **Logistic Regression** | **66.65%** | 71.42% | ±0.46% |

> **Best overall model:** Logistic Regression (66.65% test accuracy on 20 categories)

### Per-Category F1 Scores (Logistic Regression, Full Dataset)

| Best Category | F1 | Worst Category | F1 |
|---------------|----|----------------|----|
| `rec.sport.hockey` | **0.879** | `talk.religion.misc` | **0.286** |


## Visualizations

All charts are saved as `.png` files in the repository root.

| File | Description |
|------|-------------|
| `model_comparison.png` | Accuracy comparison across all three models |
| `small_vs_full_comparison.png` | Performance drop from 4 to 20 categories |
| `confusion_matrix.png` | Full 20×20 confusion matrix (Logistic Regression) |
| `f1_per_category.png` | F1 score per category bar chart |
| `top_words.png` | Top 20 most important words by mean TF-IDF score |
| `top_words_per_category.png` | Top 10 discriminative words per category (Naive Bayes log-probs) |


## Repository Structure

```
├── topic_modeling_nlp.ipynb       # Main notebook — full pipeline
├── confusion_matrix.png
├── f1_per_category.png
├── model_comparison.png
├── small_vs_full_comparison.png
├── top_words.png
├── top_words_per_category.png
├── saved_models/
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   ├── logistic_regression.pkl
│   └── tfidf_vectorizer.pkl
└── README.md
```


## Setup & Usage

### Requirements

```bash
pip install scikit-learn nltk pandas numpy matplotlib seaborn joblib
```

### NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Run the Notebook

Open `topic_modeling_nlp.ipynb` and run all cells in order. The notebook will automatically download the 20 Newsgroups dataset via scikit-learn.

### Load a Saved Model

```python
import joblib

model = joblib.load('saved_models/logistic_regression.pkl')
vec   = joblib.load('saved_models/tfidf_vectorizer.pkl')

# Predict on new text
text = ["The launch of the new space telescope was a success."]
pred = model.predict(vec.transform(text))
print(pred)  # e.g. [14]  →  sci.space
```


## Key Findings

- **Accuracy drops sharply** (~22 pp) moving from 4 to 20 categories - expected as topics become more semantically overlapping.
- **Most confused pairs** share vocabulary naturally: `talk.politics.misc` ↔ `talk.politics.guns` (88 errors), `talk.religion.misc` ↔ `soc.religion.christian` (63 errors), and `alt.atheism` ↔ `soc.religion.christian` (54 errors).
- **Best-separated topics** are domain-specific with unique vocabulary: `rec.sport.hockey` (F1 = 0.88), `misc.forsale` (F1 = 0.77), `rec.sport.baseball` (F1 = 0.79).
- **TF-IDF > CountVectorizer** by 1.73% on Naive Bayes - downweighting common cross-category words meaningfully helps.
- All three models perform within ~2% of each other on the full dataset, suggesting the ceiling is largely a feature/data constraint rather than a model choice issue.


## Tools & Libraries

`Python` · `scikit-learn` · `NLTK` · `pandas` · `NumPy` · `matplotlib` · `seaborn` · `joblib`
