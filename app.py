import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧬 Breast Cancer Classification System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("Configuration")
    st.markdown("Predict breast cancer diagnosis using advanced ML models.")
    
    st.subheader("1. Select Model")
    model_options = [
        "Logistic Regression",
        "Decision Tree",
        "k-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
    selected_model_name = st.selectbox("Choose a classifier:", model_options)
    
    st.subheader("2. Upload Data")
    uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])
    
    st.markdown("---")
    st.info("Don't have data?")
    
    # Download button for test data
    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as f:
            st.download_button(
                label="📥 Download Sample Test Data",
                data=f,
                file_name="test_data.csv",
                mime="text/csv"
            )

# Load Model
@st.cache_resource
def load_model(model_name):
    filename = f"model/{model_name.replace(' ', '_').lower()}.pkl"
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        st.error(f"⚠️ Model file `{filename}` not found. Please run Training Script first.")
        return None

# Load Scaler
@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('model/scaler.pkl')
        return scaler
    except FileNotFoundError:
        st.error("⚠️ Scaler file `model/scaler.pkl` not found.")
        return None

model = load_model(selected_model_name)
scaler = load_scaler()

# Main Content
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📊 Data Preview")
            st.write(f"Uploaded {data.shape[0]} rows, {data.shape[1]} columns")
            st.dataframe(data.head(10), height=300)
        
        # Check for target column
        if 'target' in data.columns:
            y_true = data['target']
            X = data.drop('target', axis=1)
            has_target = True
        else:
            X = data
            has_target = False
            st.warning("⚠️ No 'target' column found. Evaluation metrics will be skipped.")

        if model is not None and scaler is not None:
            # Predict
            try:
                X_scaled = scaler.transform(X)
                y_pred = model.predict(X_scaled)
                y_prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None
                
                # Results
                results_df = X.copy()
                results_df['Prediction'] = y_pred
                target_map = {0: "Malignant", 1: "Benign"} # Check mapping!
                # RE-CHECK MAPPING: In sklearn cancer dataset: 0=Malignant, 1=Benign.
                # However, usually 1 is the 'positive' class (Malignant) for metrics like Recall/Precision.
                # Let's check the script generation logic. 
                # In script: 
                # model.fit(X_train, y_train) 
                # data = load_breast_cancer()
                # Class distribution: 212 - Malignant, 357 - Benign
                # target_names: ['malignant', 'benign'] -> 0, 1
                # So 0 is malignant, 1 is benign.
                # This is standard sklearn.
                
                results_df['Diagnosis'] = results_df['Prediction'].map(target_map)
                
                with col2:
                    st.subheader("🔮 Predictions")
                    st.dataframe(results_df[['Diagnosis'] + list(X.columns)].head(10), height=300)

                if has_target:
                    st.markdown("---")
                    st.header("📈 Evaluation Metrics")
                    
                    # Calculate Metrics
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred)
                    rec = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    mcc = matthews_corrcoef(y_true, y_pred)
                    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0

                    # Metrics Row
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Accuracy", f"{acc:.2%}")
                    m2.metric("AUC", f"{auc:.3f}")
                    m3.metric("Precision", f"{prec:.3f}")
                    m4.metric("Recall", f"{rec:.3f}")
                    m5.metric("F1 Score", f"{f1:.3f}")
                    m6.metric("MCC", f"{mcc:.3f}")
                    
                    # Plots
                    st.markdown("---")
                    p1, p2 = st.columns(2)
                    
                    with p1:
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                    xticklabels=['Malignant', 'Benign'],
                                    yticklabels=['Malignant', 'Benign'])
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                        
                    with p2:
                        st.subheader("Classification Report")
                        report = classification_report(y_true, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.highlight_max(axis=0), height=300)
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
                
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

else:
    st.markdown("""
    ### 👋 Welcome!
    Please upload a CSV file from the sidebar to start using the system.
    
    **Features:**
    - Support for multiple advanced classifiers.
    - Real-time prediction and evaluation.
    - Interactive visualizations.
    """)
    if os.path.exists("test_data.csv"):
        st.success("💡 Tip: Download the sample test data from the sidebar to test immediately.")

st.markdown("---")
st.markdown("© 2026 M.Tech Machine Learning Assignment 2")
