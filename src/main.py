import os
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import preprocess_data, vectorize_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def main():
    """
    Main function to run the spam classification pipeline.
    This script trains a model, evaluates it, and saves all necessary artifacts.
    """
    # --- 1. Setup and Directory Creation ---
    print("Setting up directories...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('assets', exist_ok=True)

    # --- 2. Data Loading and Preprocessing ---
    print("Loading and preprocessing data...")
    df = pd.read_csv('dataset/sms_spam.csv', header=None, names=['label', 'text'], encoding='latin-1')
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df = preprocess_data(df, text_column='text')

    # --- 3. Text Vectorization ---
    print("Vectorizing text data...")
    corpus = df['cleaned_text'].tolist()
    X, vectorizer = vectorize_text(corpus)

    # --- 4. Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

    # --- 5. Model Training ---
    print("Training logistic regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully!")

    # --- 6. Model Evaluation ---
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # --- 7. Artifact Serialization ---
    print("Saving artifacts...")

    # Save model and vectorizer
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    print("  Model and vectorizer saved to 'models/'")

    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    with open('assets/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("  Metrics saved to 'assets/metrics.json'")

    # Create and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('assets/confusion_matrix.png')
    print("  Confusion matrix plot saved to 'assets/confusion_matrix.png'")
    
    print("\nPipelin`e finished successfully.")


if __name__ == '__main__':
    main()