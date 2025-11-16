import re
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def clean_text(text):
    """
    Cleans the input text by converting it to lowercase and removing punctuation.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text

def tokenize_text(text):
    """
    Tokenizes the input text.

    Args:
        text (str): The input text.

    Returns:
        list: A list of tokens.
    """
    return word_tokenize(text)

def vectorize_text(data):
    """
    Vectorizes the input data using TF-IDF.

    Args:
        data (list): A list of documents (strings).

    Returns:
        tuple: A tuple containing the vectorized data and the TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)
    return X, vectorizer

def preprocess_data(df, text_column='text'):
    """
    Applies the full preprocessing pipeline to a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str, optional): The name of the column containing the text to preprocess. 
                                     Defaults to 'text'.

    Returns:
        pd.DataFrame: The DataFrame with preprocessed text.
    """
    df['cleaned_text'] = df[text_column].apply(clean_text)
    df['tokenized_text'] = df['cleaned_text'].apply(tokenize_text)
    return df

if __name__ == '__main__':
    # Create a sample DataFrame for demonstration
    data = {'text': ["This is the first document.",
                     "This document is the second document.",
                     "And this is the third one.",
                     "Is this the first document?"]}
    df = pd.DataFrame(data)

    # Preprocess the data
    df = preprocess_data(df)
    print("Preprocessed DataFrame:")
    print(df)

    # Vectorize the data
    corpus = df['cleaned_text'].tolist()
    X, vectorizer = vectorize_text(corpus)

    print("\nFeature names:")
    print(vectorizer.get_feature_names_out())

    print("\nTF-IDF Vectorized Data (sparse matrix):")
    print(X.toarray())
