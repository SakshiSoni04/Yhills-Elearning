# collaborative.py
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from models.models import DatabaseManager


def collaborative_recommendation(courses: pd.DataFrame, user_ratings: dict, top_n: int = 10):
    """
    Enhanced collaborative filtering with matrix factorization (SVD)
    """
    # Get all ratings from database
    db = DatabaseManager()
    all_ratings = db.get_all_ratings()

    if all_ratings.empty or len(all_ratings) < 10:
        return collaborative_fallback(courses, user_ratings, top_n)

    # Create user-item matrix
    user_item_matrix = all_ratings.pivot_table(
        index='username', columns='course_id', values='value'
    ).fillna(0)

    # Convert to numpy array
    ratings_matrix = user_item_matrix.values

    # Normalize by each user's mean
    user_ratings_mean = np.mean(ratings_matrix, axis=1)
    ratings_normalized = ratings_matrix - user_ratings_mean.reshape(-1, 1)

    # Perform SVD
    U, sigma, Vt = svds(ratings_normalized, k=min(50, len(user_item_matrix.columns) - 1))
    sigma = np.diag(sigma)

    # Make predictions
    all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_predicted_ratings,
                            columns=user_item_matrix.columns,
                            index=user_item_matrix.index)

    # Get recommendations for current user
    if user_ratings:
        user_vector = pd.DataFrame(0, index=['current_user'], columns=user_item_matrix.columns)
        for course_id, rating in user_ratings.items():
            if course_id in user_vector.columns:
                user_vector.loc['current_user', course_id] = rating

        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(user_vector, user_item_matrix)
        similar_users = np.argsort(similarity[0])[-5:]

        user_preds = preds_df.iloc[similar_users].mean(axis=0)
    else:
        user_preds = preds_df.mean(axis=0)

    user_preds = user_preds[~user_preds.index.isin(user_ratings.keys())]
    top_course_ids = user_preds.sort_values(ascending=False).head(top_n).index.tolist()

    recommendations = courses[courses['Course_ID'].isin(top_course_ids)].copy()
    recommendations['Predicted_Rating'] = recommendations['Course_ID'].map(user_preds)

    return recommendations.sort_values('Predicted_Rating', ascending=False).head(top_n)


def collaborative_fallback(courses: pd.DataFrame, user_ratings: dict, top_n: int = 10):
    """Fallback collaborative filtering"""
    rated_titles = set(user_ratings.keys())
    ranked = courses.sort_values(by=["Rate", "Reviews"], ascending=[False, False])
    ranked = ranked[~ranked["Course_ID"].isin(rated_titles)]
    return ranked.head(top_n)