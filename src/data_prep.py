from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_tfidf(df):
    """Build TF-IDF matrix using Gained Skills + Title + Subject text."""
    df["combined_text"] = (
        df["Title"].fillna("") + " " +
        df["Subject"].fillna("") + " " +
        df["Gained Skills"].fillna("")
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    return vectorizer, tfidf_matrix