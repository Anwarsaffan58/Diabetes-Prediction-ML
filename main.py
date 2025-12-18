"""
PROJECT: Diabetes Prediction System
AUTHOR: Saffan
DESCRIPTION: 
    A Supervised Learning pipeline (Random Forest) to predict diabetes probability.
    Includes data synthesis, preprocessing, model training, and feature importance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'data_path': 'data/diabetes_data.csv',
    'matrix_path': 'outputs/confusion_matrix.png',
    'importance_path': 'outputs/feature_importance.png',
    'report_path': 'outputs/prediction_report.csv',
    'n_samples': 1000,
    'test_size': 0.2,
    'random_state': 42
}

def ensure_directories():
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

# ==========================================
# PART 1: DATA GENERATION (Synthetic Medical Data)
# ==========================================
def generate_medical_data():
    """
    Generates a realistic dataset mimicking the Pima Indians Diabetes Database.
    Correlations: High Glucose & BMI -> Higher chance of Outcome=1.
    """
    print("[1/5] Generating synthetic medical records...")
    
    np.random.seed(CONFIG['random_state'])
    n = CONFIG['n_samples']
    
    # Generate Features
    age = np.random.randint(21, 81, n)
    pregnancies = np.random.randint(0, 15, n)
    
    # Glucose: Diabetics tend to have higher glucose
    glucose = np.random.normal(100, 20, n) 
    
    # BMI: Diabetics tend to have higher BMI
    bmi = np.random.normal(30, 6, n)
    
    blood_pressure = np.random.normal(70, 10, n)
    insulin = np.random.normal(80, 20, n)
    pedigree = np.abs(np.random.normal(0.5, 0.3, n)) # Genetic score
    
    # Create the Target (0 = No Diabetes, 1 = Diabetes)
    # Logic: If Glucose > 140 or BMI > 35, higher probability of diabetes
    risk_score = (glucose * 0.4) + (bmi * 0.3) + (age * 0.1) + (pedigree * 10) + np.random.normal(0, 10, n)
    threshold = np.percentile(risk_score, 70) # Top 30% are diabetic
    outcome = (risk_score > threshold).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': np.random.randint(20, 50, n),
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': pedigree,
        'Age': age,
        'Outcome': outcome
    })
    
    # Save raw data
    df.to_csv(CONFIG['data_path'], index=False)
    print(f"      Dataset saved to {CONFIG['data_path']}")
    return df

# ==========================================
# PART 2: MODEL TRAINING (Random Forest)
# ==========================================
def train_model(df):
    """
    Trains a Random Forest Classifier to predict 'Outcome'.
    """
    print("[2/5] Training Random Forest model...")
    
    # Features (X) and Target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
    )
    
    # Initialize Model (Optimized parameters)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=CONFIG['random_state']
    )
    
    # Fit Model
    model.fit(X_train, y_train)
    
    # Make Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"      Model Accuracy: {acc*100:.2f}%")
    
    return model, X_test, y_test, y_pred, X.columns

# ==========================================
# PART 3: VISUALIZATION & ANALYSIS
# ==========================================
def evaluate_and_visualize(model, X_test, y_test, y_pred, feature_names):
    """
    Generates Confusion Matrix and Feature Importance plots.
    """
    print("[3/5] Generating Confusion Matrix...")
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Actual vs Predicted)')
    plt.xlabel('Predicted Label (0=Healthy, 1=Diabetic)')
    plt.ylabel('Actual Label')
    plt.savefig(CONFIG['matrix_path'])
    plt.close()
    
    print("[4/5] Analyzing Feature Importance...")
    
    # 2. Feature Importance Plot
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
    plt.title('Key Predictors of Diabetes Risk')
    plt.xlabel('Importance Score')
    plt.ylabel('Medical Features')
    plt.tight_layout()
    plt.savefig(CONFIG['importance_path'])
    plt.close()
    
    print(f"      Visualizations saved to outputs/")

def save_predictions(X_test, y_test, y_pred):
    """Saves the test results to a CSV."""
    print("[5/5] Saving prediction report...")
    
    results = X_test.copy()
    results['Actual_Outcome'] = y_test
    results['Predicted_Outcome'] = y_pred
    
    # Filter only the incorrect predictions for analysis
    results.to_csv(CONFIG['report_path'], index=False)
    print(f"      Report saved to {CONFIG['report_path']}")

# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    print("--- STARTING DIABETES PREDICTION PIPELINE ---\n")
    
    ensure_directories()
    
    # 1. Data
    data = generate_medical_data()
    
    # 2. Train
    model, X_test, y_test, y_pred, feature_cols = train_model(data)
    
    # 3. Visualize
    evaluate_and_visualize(model, X_test, y_test, y_pred, feature_cols)
    
    # 4. Save
    save_predictions(X_test, y_test, y_pred)
    
    print("\n--- PIPELINE COMPLETE ---")