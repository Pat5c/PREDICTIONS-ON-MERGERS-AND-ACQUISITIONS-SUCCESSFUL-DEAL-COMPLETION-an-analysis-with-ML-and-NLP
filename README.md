# Predicting M&A Deal Completion Using Machine Learning and NLP

A Data Science project that applies **Machine Learning (ML)** and **Natural Language Processing (NLP)** to predict the successful completion of **Mergers & Acquisitions (M&A)** deals. This work explores risk arbitrage strategies by integrating structured financial data and unstructured textual sentiment extracted from headlines and news before deal announcements: by combining structured financial datasets with sentiment analysis of pre-announcement headlines and rumors, I developed a hybrid model that improves forecasting accuracy for M&A deal outcomes.

![MandA_Picture](https://github.com/user-attachments/assets/ce849b0b-2364-43b5-9cad-b83642528e54)



---

## Project Objectives

This project aims to:

1. **Predict the success of M&A deal completion** using various ML classification models.
2. **Identify the best-performing model** against a Logistic Regression baseline.
3. **Forecast imminent deal announcements** via sentiment analysis of news prior to announcements.
4. **Assess whether combining sentiment data with financial variables improves predictions.**

---

## Background & Motivation

M&A deals play a major role in financial markets, yet misclassifying an unsuccessful deal as successful can lead to significant losses for arbitrageurs and investors. 

By analyzing both numerical (financial) and textual (news) datasets, this project attempts to:
- Predict deal completion.
- Analyze rumor-driven stock price movements.
- Understand how sentiment and financial signals interact.

---

## Technologies Used

- **Languages:** Python
- **ML Libraries:** `scikit-learn`, `xgboost`, `keras`
- **NLP Tools:** `VADER`, `TextBlob`, `NLTK`, `spaCy`
- **EDA & Viz:** `pandas`, `matplotlib`, `seaborn`, `plotly`
- **Model Evaluation:** Precision, Recall, F1-Score, Accuracy

---
## Table of Contents

DEDICATION

ACKNOWLEDGMENTS

ABSTRACT

TABLE OF CONTENTS

LIST OF TABLES

LIST OF FIGURES

LIST OF ABBREVIATIONS

CHAPTER 1 - INTRODUCTION

Introduction  

  Research rationale, aims and objectives 
  
  Dissertation structure 
  
CHAPTER 2 - BACKGROUND LITERATURE

DATASETS

  Data Pre-Processing and Cleaning
  
  Variables selection 
  
RESULTS & EVALUATION - Second research question answered

CHAPTER 3 - METHODOLOGIES

METHODOLOGY 1 – Models with No sentiment scores 

  Baseline model: Logistic Regression 
  
  Other classification models 
  
  Models’ evaluation metrics 
  
RESULTS & EVALUATION – First research question answered 

METHODOLOGY II – Models with sentiment score 

  Sentiment analysis with TextBlob: Baseline Model
  
  Sentiment analysis with VADER
  
Approach A: Share price with sentiment scores

  RESULTS & EVALUATION – Third research question answered 
  
Approach B: Financial dataset with sentiment scores 

  RESULTS & EVALUATION – Fourth and last research question answered 
  
CONCLUSIONS

FURTHER WORK

REFERENCES 

APPENDICES

  Appendix A: Datasets Processing & Cleaning Code 
  
  Appendix B: Variables Selection
  
  Appendix C: METHODOLOGY I – Models with No sentiment scores 
  
  Appendix D: METHODOLOGY II – Models with sentiment score
  
---

## Methodology Overview

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

## Key Research Questions

1. Can models like SVM or Random Forest outperform Logistic Regression in predicting M&A deal success?
2. Which financial variables most influence M&A success prediction?
3. Can pre-deal sentiment signals + price trends predict imminent M&A announcements?
4. Does combining sentiment scores with financial features improve prediction accuracy?

---

## Results Summary

### ✅ Best Performing Models (Financial Only):
- **Random Forest**, **SVM**, and **Decision Tree** outperformed Logistic Regression.
- All achieved **100% Precision, Recall, F1, and Accuracy**.
- Logistic Regression scored slightly lower in all metrics.

### 💬 Sentiment Analysis Results:
- Rumor-driven headlines linked to significant share price run-ups pre-announcement.
- VADER and TextBlob effectively captured sentiment dynamics.
- Hypothesis confirmed: **High prices + low positive sentiment** → Strong signal for imminent M&A.

  <img width="380" height="250" alt="image" src="https://github.com/user-attachments/assets/11717f2b-3d29-4399-8c03-07f6b6b89b9f" />


### 🧪 Combined (NLP + Financial):
- Sentiment scores **did not degrade** model performance.
- Models like Random Forest and Decision Tree retained 100% metrics.
- No statistically significant improvement, but strong evidence for **complementary predictive signals**.

<img width="465" height="178" alt="image" src="https://github.com/user-attachments/assets/b5652324-6bef-42c8-9d0f-d58b0c3b32b0" />

---

THANK YOU!

If you find this project interesting, feel free to connect on LinkedIn.

