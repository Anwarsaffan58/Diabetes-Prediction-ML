# 🏥 Diabetes Risk Predictor

A healthcare machine learning model designed to predict the likelihood of diabetes in patients based on diagnostic metrics.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/Model-RandomForest-orange)

## 🚀 Project Overview
Early detection of diabetes can significantly improve patient outcomes. This project implements a **Supervised Learning** pipeline that analyzes patient health data (Glucose, BMI, Insulin levels, etc.) to classify individuals as "Low Risk" or "High Risk."

It leverages **Feature Importance Analysis** to identify which medical factors contribute most to the diagnosis.

### 🔑 Key Features
- **Algorithm**: Uses **Random Forest Classifier** for high accuracy and interpretability.
- **Feature Optimization**: Automatically ranks medical factors (e.g., Glucose vs. BP) to determine predictive power.
- **Synthetic Medical Data**: Generates realistic patient records (1,000 samples) based on clinical correlations.
- **Diagnostic Visualizations**: Outputs Confusion Matrices and Feature Importance charts.

## 🛠️ Tech Stack
- **Python**: Core logic.
- **Scikit-Learn**: Model training and evaluation.
- **Pandas/NumPy**: Data manipulation.
- **Seaborn/Matplotlib**: Medical data visualization.

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/diabetes-predictor.git](https://github.com/yourusername/diabetes-predictor.git)
cd diabetes-predictor
