# 📧 Email Spam Classifier (HW3)

This project builds a machine-learning model to classify SMS messages as Spam or Ham (Not Spam). It reproduces the core concepts from Chapter 3 of "Hands-On AI for Cybersecurity" while modernizing the workflow with OpenSpec for spec-driven development, NLTK for robust preprocessing, and a Streamlit web interface for interactive predictions.

---

## 📂 Project Structure

```
HW3_EmailSpamProject/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── README.md               # This file
├── dataset/
│   └── sms_spam.csv        # Training data
├── src/
│   ├── main.py             # Training and evaluation script
│   └── preprocessing.py    # Data cleaning and NLTK logic
├── models/                 # Saved model and vectorizer
│   ├── model.joblib
│   └── vectorizer.joblib
├── assets/                 # Saved evaluation artifacts
│   ├── confusion_matrix.png
│   └── metrics.json
└── openspec/               # OpenSpec for spec-driven development
    ├── AGENTS.md           # Virtual team roles
    ├── project.md          # Project conventions and context
    └── changes/            # Change proposals and specifications
```

---

## ⚡ Quick Start

Follow these steps to get the application running locally.

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then install the required packages.
```shell
pip install -r requirements.txt
```

### 2. Train the Model (Crucial Step)
Before running the app, you must run the training script. This will train the model, evaluate it, and save all the necessary artifacts (`.joblib`, `.png`, `.json`) into the `models/` and `assets/` directories.
```shell
python src/main.py
```

### 3. Run the Streamlit App
Once the training is complete and the artifacts are generated, you can launch the interactive web application.
```shell
streamlit run app.py
```
Your browser should automatically open to the application's URL.

---

## 🔍 Observations & Limitations

During testing, we observed that the model performs well on traditional spam keywords (e.g., 'Winner', 'Free', 'Urgent'). However, it produced **False Negatives** on modern scam patterns, specifically:

1.  **Package Delivery Scams:** (e.g., 'We tried to deliver your package...') - The model classified this as Ham. This is likely because the training dataset (SMS Spam Collection) is older (~2012) and predates the rise of delivery phishing texts.
2.  **Loan Offers:** Some business-like loan offers were misclassified as Ham due to neutral phrasing.

**Future Improvement:** To fix this, we would need to curate a more modern dataset containing recent phishing patterns or use contextual models like BERT.