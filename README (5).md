# Accident Severity Prediction using Machine Learning

Predicts the severity of road accidents from pre-crash environmental and situational signals using machine learning.

## Overview

This project analyzes 500,000 real US road accident records to build a classification model that predicts whether an accident will result in minor or severe impact before it occurs. The goal is to identify which pre-crash signals most strongly predict accident severity, with direct applications in autonomous vehicle safety systems and smart traffic management.

## Dataset

US Accidents dataset by Sobhan Moosavi
Source: kaggle.com/datasets/sobhanmoosavi/us-accidents
Size: 7.7 million records across 49 US states from 2016 to 2023
Features: 46 columns including weather, road features, time of day, and severity

## Tech Stack

Python, Scikit-learn, XGBoost, SHAP, SMOTE, Pandas, Matplotlib, Seaborn

## Pipeline

1. Data loading using Kaggle API
2. Exploratory Data Analysis
3. Data preprocessing and feature selection
4. Feature engineering (visibility risk, weather risk score, road complexity, wind risk)
5. Class imbalance handling with SMOTE
6. Model training and comparison (Logistic Regression, Random Forest, XGBoost)
7. Model evaluation with AUC-ROC, ROC curve, and confusion matrix
8. SHAP explainability to identify top pre-crash signals

## Results

Best Model: Random Forest
AUC-ROC: 0.7577
Features: 21 engineered pre-crash features
Training samples after SMOTE: 44,698
Test samples: 9,247

## Key Findings

SHAP analysis identified the pre-crash signals that most strongly predict severe accidents including visibility conditions, time of day, weather severity, and road complexity features.

## How to Run

1. Open accident_severity_prediction.ipynb in Google Colab
2. Upload your kaggle.json API token when prompted in Phase 3
3. Run all cells from top to bottom

## Author

Aneesa Alam
GitHub: github.com/Aneesaalam14
