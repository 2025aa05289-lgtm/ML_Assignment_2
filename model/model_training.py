import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"Warning: XGBoost not available ({e}). Using sklearn GradientBoostingClassifier instead.")
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import joblib
import os
import json

# Set random seed for reproducibility
RANDOM_STATE = 42

def load_data():
    """Load Breast Cancer dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

def preprocess_data(X, y):
    """Split and scale data."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for app
    os.makedirs('model', exist_ok=True)
    joblib.dump(scaler, 'model/scaler.pkl')
    
    # Save test data for user download/testing
    test_data = pd.DataFrame(X_test, columns=X.columns)
    test_data['target'] = y_test.values
    test_data.to_csv('test_data.csv', index=False)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train models and evaluate metrics."""
    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "k-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE) if XGBOOST_AVAILABLE else GradientBoostingClassifier(random_state=RANDOM_STATE)
    }
    
    results = {}
    
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results[name] = {
            "Accuracy": acc,
            "AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "MCC": mcc
        }
        
        # Save model
        filename = f"model/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, filename)
    
    return results

def main():
    print("Loading data...")
    X, y = load_data()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=4))
    
    # Save results to file for README generation
    with open("model_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nDone! Models saved in 'model/' directory.")

if __name__ == "__main__":
    main()
