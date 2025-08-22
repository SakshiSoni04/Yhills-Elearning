# src/content_based.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def content_based_recommendation(courses_df: pd.DataFrame, user_skills: str, top_n: int = 10):
    """
    Generate recommendations using content-based filtering (TF-IDF on Gained Skills).

    Args:
        courses_df (pd.DataFrame): Dataset of courses
        user_skills (str): Comma separated skills/interests (e.g. "python, data science")
        top_n (int): Number of recommendations

    Returns:
        pd.DataFrame: Top N recommended courses
    """

    # Fill missing skills with empty string
    courses_df["Gained Skills"] = courses_df["Gained Skills"].fillna("")

    # TF-IDF on course skills
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(courses_df["Gained Skills"])

    # Convert user skills into vector
    user_vec = tfidf.transform([user_skills])

    # Compute similarity
    cosine_sim = linear_kernel(user_vec, tfidf_matrix).flatten()

    # Rank courses
    courses_df["similarity"] = cosine_sim
    recommendations = courses_df.sort_values("similarity", ascending=False).head(top_n)

    return recommendations