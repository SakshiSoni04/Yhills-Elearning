# content_based.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np
from models.models import DatabaseManager


def content_based_recommendation(courses_df: pd.DataFrame, user_skills: str, user_id: int = None, top_n: int = 10):
    """
    Enhanced content-based filtering with LSA
    """
    # Create a working copy and reset index
    courses_working = courses_df.copy().reset_index(drop=True)

    courses_working["Gained Skills"] = courses_working["Gained Skills"].fillna("")

    courses_working["combined_text"] = (
            courses_working["Title"].fillna("") + " " +
            courses_working["Subject"].fillna("") + " " +
            courses_working["Gained Skills"].fillna("") + " " +
            courses_working["Level"].fillna("") + " " +
            courses_working["Institution"].fillna("")
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    svd = TruncatedSVD(n_components=100, random_state=42)
    lsa_pipeline = Pipeline([('tfidf', tfidf), ('svd', svd)])

    course_vectors = lsa_pipeline.fit_transform(courses_working["combined_text"])

    if user_id:
        db = DatabaseManager()
        user_ratings = db.get_user_ratings(user_id)

        if user_ratings:
            # Find which courses in the current filtered set the user has rated
            rated_mask = courses_working['Course_ID'].isin(user_ratings.keys())

            if rated_mask.any():
                # Get the indices of rated courses in the current filtered set
                rated_indices = np.where(rated_mask)[0]
                rated_course_ids = courses_working.loc[rated_mask, 'Course_ID']
                user_weights = [user_ratings[course_id] for course_id in rated_course_ids]

                # FIX: Check if weights sum to zero before using weighted average
                weights_sum = sum(user_weights)
                if weights_sum == 0:
                    # If all ratings are zero, use simple average instead
                    user_vector = np.mean(course_vectors[rated_indices], axis=0)
                else:
                    # Use weighted average for non-zero weights
                    user_vector = np.average(
                        course_vectors[rated_indices],
                        axis=0,
                        weights=user_weights
                    )
            else:
                user_vector = create_user_vector_from_skills(lsa_pipeline, user_skills)
        else:
            user_vector = create_user_vector_from_skills(lsa_pipeline, user_skills)
    else:
        user_vector = create_user_vector_from_skills(lsa_pipeline, user_skills)

    cosine_sim = cosine_similarity([user_vector], course_vectors).flatten()
    courses_working["similarity"] = cosine_sim
    recommendations = courses_working.sort_values("similarity", ascending=False).head(top_n)

    return recommendations


def create_user_vector_from_skills(lsa_pipeline, user_skills: str):
    user_vec = lsa_pipeline.named_steps['tfidf'].transform([user_skills])
    user_vec = lsa_pipeline.named_steps['svd'].transform(user_vec)
    return user_vec.flatten()