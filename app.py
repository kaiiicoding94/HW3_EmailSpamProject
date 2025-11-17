import streamlit as st
import joblib
import json
import os
from src.preprocessing import clean_text

# --- Page Configuration ---
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# --- Artifact Paths ---
MODEL_PATH = 'models/model.joblib'
VECTORIZER_PATH = 'models/vectorizer.joblib'
METRICS_PATH = 'assets/metrics.json'
CM_PLOT_PATH = 'assets/confusion_matrix.png'

# --- Helper Function to Load Artifacts ---
@st.cache_resource
def load_artifacts():
    """
    Loads the trained model, vectorizer, and performance metrics.
    Returns None for any artifact that fails to load.
    """
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        model = None

    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        vectorizer = None
        
    return model, vectorizer

def check_artifacts_exist():
    """Checks if all necessary artifact files exist."""
    return all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, METRICS_PATH, CM_PLOT_PATH])

# --- Main Application ---

# --- Header ---
st.title("📧 Email Spam Classifier (HW3)")
st.markdown("""
This application uses a trained Logistic Regression model to classify email messages as **Spam** or **Ham** (not spam). 
Enter a message below and click 'Predict' to see the result.
""")

# --- Artifact Check ---
if not check_artifacts_exist():
    st.error("""
    **Model artifacts not found!** 
    
    Please run the training script first to generate the necessary model and evaluation files:
    ```
    python src/main.py
    ```
    After the script finishes, refresh this page.
    """)
else:
    # --- Load Model and Vectorizer ---
    model, vectorizer = load_artifacts()

    # --- Prediction UI ---
    st.header("Classify a Message")
    user_input = st.text_area("Enter the email message text here:", height=150)

    if st.button("Predict"):
        if user_input.strip() and model and vectorizer:
            # 1. Preprocess the input text
            cleaned_input = clean_text(user_input)
            
            # 2. Vectorize the cleaned text
            vectorized_input = vectorizer.transform([cleaned_input])
            
            # 3. Predict using the model
            prediction = model.predict(vectorized_input)
            prediction_proba = model.predict_proba(vectorized_input)

            # 4. Display the result
            result = "Spam" if prediction[0] == 1 else "Ham"
            probability = prediction_proba[0][prediction[0]]

            if result == "Spam":
                st.warning(f"**Result: {result}** (Confidence: {probability:.2%})")
            else:
                st.success(f"**Result: {result}** (Confidence: {probability:.2%})")
        elif not user_input.strip():
            st.info("Please enter some text to classify.")

    # --- Model Performance Display ---
    st.divider()
    st.header("Model Performance")

    try:
        # Load and display metrics
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", f"{metrics['precision']:.4f}")
        col2.metric("Recall", f"{metrics['recall']:.4f}")
        col3.metric("F1-Score", f"{metrics['f1_score']:.4f}")

        # Display confusion matrix
        st.subheader("Confusion Matrix")
        st.image(CM_PLOT_PATH, caption="Confusion matrix from the test set.")

    except FileNotFoundError:
        # This case is handled by the main artifact check, but included for robustness
        st.warning("Could not load performance files.")
