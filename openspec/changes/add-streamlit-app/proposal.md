# Change: Add Streamlit Visualization App

## Why
To fulfill the "Visualization & Streamlit" project requirement, which is currently missing and accounts for 20% of the total grade. This change introduces a user-facing web application for interacting with the spam classification model.

## What Changes
- **Modify `src/main.py`:** The existing training script will be updated to serialize and save the trained `LogisticRegression` model, the `TfidfVectorizer`, and evaluation artifacts (a confusion matrix plot and a dictionary of metrics).
- **Create `app.py`:** A new Streamlit application will be created at the root level.
- **Add UI for Prediction:** The app will provide a text area for user input and a button to classify the message as "Spam" or "Ham".
- **Add UI for Performance:** The app will display the model's performance, including the saved confusion matrix plot and key metrics like accuracy, precision, and recall.
- **Create artifact directories:** New directories `models/` and `assets/` will be created to store the outputs from the training script.

## Impact
- **Affected Specs:** `spam-classification`
- **Affected Code:**
    - `src/main.py` (will be modified)
    - `app.py` (new file)
- **Dependencies:** `streamlit`, `matplotlib`, and `seaborn` will be added to `requirements.txt`.
