## 1. Enhance Training Script (`src/main.py`)
- [ ] 1.1. Create `models/` and `assets/` directories to store artifacts.
- [ ] 1.2. Modify `src/main.py` to import `joblib`, `matplotlib`, `seaborn`, and `json`.
- [ ] 1.3. After training, save the `LogisticRegression` model to `models/model.joblib`.
- [ ] 1.4. Save the fitted `TfidfVectorizer` to `models/vectorizer.joblib`.
- [ ] 1.5. Generate a confusion matrix, plot it using `seaborn`, and save it as `assets/confusion_matrix.png`.
- [ ] 1.6. Calculate accuracy, precision, recall, and F1-score and save them as a dictionary in `assets/metrics.json`.

## 2. Create Streamlit Application (`app.py`)
- [ ] 2.1. Create a new file `app.py` in the project root.
- [ ] 2.2. Add the title "Email Spam Classifier (HW3)".
- [ ] 2.3. Load the saved model (`models/model.joblib`) and vectorizer (`models/vectorizer.joblib`).
- [ ] 2.4. Create a `st.text_area` for user input and a `st.button` labeled "Predict".
- [ ] 2.5. Implement the prediction logic: take user input, transform it using the loaded vectorizer, and predict using the loaded model.
- [ ] 2.6. Display the prediction result clearly to the user (e.g., "Result: **Spam**" or "Result: **Ham**").

## 3. Display Model Performance in App
- [ ] 3.1. In `app.py`, add a subheader for "Model Performance".
- [ ] 3.2. Load the `assets/metrics.json` file and display the metrics.
- [ ] 3.3. Load and display the `assets/confusion_matrix.png` image.

## 4. Finalize Documentation
- [ ] 4.1. Add `streamlit`, `matplotlib`, and `seaborn` to `requirements.txt`.
- [ ] 4.2. Update `README.md` with instructions on how to run the training script and the new Streamlit application.
