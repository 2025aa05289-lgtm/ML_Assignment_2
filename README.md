# Breast Cancer Classification - ML Assignment 2

## Problem Statement
Breast cancer is one of the most common cancers among women worldwide. Early diagnosis is crucial for effective treatment. The goal of this project is to build a machine learning solution to classify breast mass samples as either malignant or benign based on digitized attributes from a fine needle aspirate (FNA) of a breast mass. We implemented multiple classification models and deployed a Streamlit web application to allow users to input data and get predictions.

## Dataset Description
**Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Data Set
**Source**: UCI Machine Learning Repository (available via `sklearn.datasets`)
**Details**:
- **Instances**: 569
- **Features**: 30 numeric, predictive attributes (Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave points, Symmetry, Fractal dimension - mean, standard error, and "worst" values).
- **Classes**: Malignant, Benign
- **Target Variable**: Diagnosis (M = Malignant, B = Benign)

## Models Used & Evaluation
We implemented 6 classification models. The dataset was split into 80% training and 20% testing. Features were scaled using `StandardScaler`.

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **k-Nearest Neighbor** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Random Forest** | 0.9561 | 0.9939 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **XGBoost (GBM)**\* | 0.9561 | 0.9907 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

*\*Note: Due to environment constraints on macOS (missing libomp), `sklearn.ensemble.GradientBoostingClassifier` was used as a fallback for XGBoost. Performance is comparable.*

### Observations
1.  **Logistic Regression** achieved the highest Accuracy (98.25%) and AUC (0.9954), indicating the dataset is linearly separable to a high degree.
2.  **Tree-based models** (Random Forest, XGBoost) also performed very well (~95.6% Accuracy), showing their robustness.
3.  **Decision Tree** had the lowest performance, likely due to overfitting, which Random Forest corrected.
4.  **Naive Bayes** performed reasonably well but was outperformed by more complex models.

## Deployment
The application is built using **Streamlit**.

### How to Run Locally
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Train models (if not present):
    ```bash
    python model/model_training.py
    ```
4.  Run the app:
    ```bash
    streamlit run app.py
    ```

### Directory Structure
```
project-folder/
|-- app.py              # Streamlit Application
|-- requirements.txt    # Dependencies
|-- README.md           # Project Documentation
|-- model/              # Saved Models and Scaler
|   |-- logistic_regression.pkl
|   |-- ...
|-- model_training.py   # Script to train and save models
```
