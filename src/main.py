import pandas as pd
from preprocessing import preprocess_data, vectorize_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    """
    Main function to run the spam classification pipeline.
    """
    # Load the dataset
    df = pd.read_csv('dataset/sms_spam.csv', header=None, names=['label', 'text'], encoding='latin-1')

    print("Original DataFrame:")
    print(df.head())

    # Map labels to numerical values
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Preprocess the data
    df = preprocess_data(df, text_column='text')
    print("\nPreprocessed DataFrame:")
    print(df.head())

    # Vectorize the data
    corpus = df['cleaned_text'].tolist()
    X, vectorizer = vectorize_text(corpus)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("\nModel trained successfully!")

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    main()
