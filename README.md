# 🤖 Predicting M&A Deal Completion Using Machine Learning and NLP

A data science project that applies **machine learning (ML)** and **natural language processing (NLP)** to predict the successful completion of **Mergers & Acquisitions (M&A)** deals. This work explores risk arbitrage strategies by integrating structured financial data and unstructured textual sentiment extracted from headlines and news before deal announcements.

---

## 🧠 Project Objectives

This project aims to:

1. **Predict the success of M&A deal completion** using various ML classification models.
2. **Identify the best-performing model** against a Logistic Regression baseline.
3. **Forecast imminent deal announcements** via sentiment analysis of news prior to announcements.
4. **Assess whether combining sentiment data with financial variables improves predictions.**

---

## 🏦 Background & Motivation

M&A deals play a major role in financial markets, yet misclassifying an unsuccessful deal as successful can lead to significant losses for arbitrageurs and investors. 

By analyzing both numerical (financial) and textual (news) datasets, this project attempts to:
- Predict deal completion.
- Analyze rumor-driven stock price movements.
- Understand how sentiment and financial signals interact.

---

## 🧰 Technologies Used

- **Languages:** Python
- **ML Libraries:** `scikit-learn`, `xgboost`, `keras`
- **NLP Tools:** `VADER`, `TextBlob`, `NLTK`, `spaCy`
- **EDA & Viz:** `pandas`, `matplotlib`, `seaborn`, `plotly`
- **Model Evaluation:** Precision, Recall, F1-Score, Accuracy

---

## 🧪 Methodology Overview

### 📊 Financial Data Pipeline:
- Dimensionality reduction using:
  - Pearson correlation
  - Spearman rank
  - T-test and filtering methods
- Classification models used:
  - Logistic Regression (Baseline)
  - Random Forest
  - Decision Tree
  - Neural Networks
  - Support Vector Machine (SVM)

### 📰 Textual Data (News/Headlines Pre-Deal):
- Cleaned and tokenized headlines.
- Sentiment Analysis using:
  - VADER
  - TextBlob
- Sentiment scores evaluated:
  - With share price run-ups
  - Integrated into financial dataset

---

## 🔍 Key Research Questions

1. Can models like SVM or Random Forest outperform Logistic Regression in predicting M&A deal success?
2. Which financial variables most influence M&A success prediction?
3. Can pre-deal sentiment signals + price trends predict imminent M&A announcements?
4. Does combining sentiment scores with financial features improve prediction accuracy?

---

## 📈 Results Summary

### ✅ Best Performing Models (Financial Only):
- **Random Forest**, **SVM**, and **Decision Tree** outperformed Logistic Regression.
- All achieved **100% Precision, Recall, F1, and Accuracy**.
- Logistic Regression scored slightly lower in all metrics.

### 💬 Sentiment Analysis Results:
- Rumor-driven headlines linked to significant share price run-ups pre-announcement.
- VADER and TextBlob effectively captured sentiment dynamics.
- Hypothesis confirmed: **High prices + low positive sentiment** → Strong signal for imminent M&A.

### 🧪 Combined (NLP + Financial):
- Sentiment scores **did not degrade** model performance.
- Models like Random Forest and Decision Tree retained 100% metrics.
- No statistically significant improvement, but strong evidence for **complementary predictive signals**.

---

