# hybrid.py
import pandas as pd
from src.content_based import content_based_recommendation
from src.collaborative import collaborative_recommendation
import numpy as np


def hybrid_recommendation(courses: pd.DataFrame, user_skills: str, user_ratings: dict,
                          user_id: int = None, top_n: int = 10, alpha: float = 0.7,
                          use_ncf: bool = False):
    """
    Enhanced hybrid recommendation with weighted combination
    Supports both SVD and Neural Collaborative Filtering options
    """
    # Make a copy of the courses DataFrame to avoid modifying the original
    courses_copy = courses.copy()

    # Get content-based recommendations
    content_recs = content_based_recommendation(courses_copy, user_skills, user_id, top_n=top_n * 3)

    # Get collaborative filtering recommendations (SVD or NCF)
    collab_recs = collaborative_recommendation(courses_copy, user_ratings, top_n=top_n * 3, use_ncf=use_ncf)

    # Create a comprehensive DataFrame with all courses
    all_courses = courses_copy.copy()
    all_courses['content_score'] = 0.0
    all_courses['collab_score'] = 0.0
    all_courses['hybrid_score'] = 0.0

    # Map content-based scores with smoothing to avoid zero scores
    if not content_recs.empty:
        content_scores = dict(zip(content_recs['Course_ID'], content_recs['similarity']))
        all_courses['content_score'] = all_courses['Course_ID'].map(content_scores).fillna(0)

        # Add small value to avoid zero scores for courses not in content recommendations
        min_content_score = all_courses['content_score'][all_courses['content_score'] > 0].min() if any(
            all_courses['content_score'] > 0) else 0
        all_courses['content_score'] = all_courses['content_score'] + (
            min_content_score * 0.1 if min_content_score > 0 else 0.01)

    # Map collaborative filtering scores with smoothing
    if not collab_recs.empty:
        if 'Predicted_Rating' in collab_recs.columns:
            collab_scores = dict(zip(collab_recs['Course_ID'], collab_recs['Predicted_Rating']))
        elif 'similarity' in collab_recs.columns:  # Fallback to similarity if Predicted_Rating doesn't exist
            collab_scores = dict(zip(collab_recs['Course_ID'], collab_recs['similarity']))
        else:
            collab_scores = {}

        all_courses['collab_score'] = all_courses['Course_ID'].map(collab_scores).fillna(0)

        # Add small value to avoid zero scores for courses not in collaborative recommendations
        min_collab_score = all_courses['collab_score'][all_courses['collab_score'] > 0].min() if any(
            all_courses['collab_score'] > 0) else 0
        all_courses['collab_score'] = all_courses['collab_score'] + (
            min_collab_score * 0.1 if min_collab_score > 0 else 0.01)

    # Normalize scores to [0, 1] range
    content_max = all_courses['content_score'].max()
    collab_max = all_courses['collab_score'].max()

    if content_max > 0:
        all_courses['content_score'] = all_courses['content_score'] / content_max

    if collab_max > 0:
        all_courses['collab_score'] = all_courses['collab_score'] / collab_max

    # Calculate hybrid score with weighted combination
    all_courses['hybrid_score'] = (
            alpha * all_courses['content_score'] +
            (1 - alpha) * all_courses['collab_score']
    )

    # Apply popularity boost for courses with high ratings and reviews
    if 'Rate' in all_courses.columns and 'Reviews' in all_courses.columns:
        # Normalize ratings and reviews
        max_rate = all_courses['Rate'].max() if all_courses['Rate'].max() > 0 else 1
        max_reviews = all_courses['Reviews'].max() if all_courses['Reviews'].max() > 0 else 1

        rating_norm = all_courses['Rate'] / max_rate
        reviews_norm = np.log1p(all_courses['Reviews']) / np.log1p(max_reviews)  # Log scaling for reviews

        # Apply popularity boost (up to 10% increase)
        popularity_boost = 0.1 * (0.7 * rating_norm + 0.3 * reviews_norm)
        all_courses['hybrid_score'] = all_courses['hybrid_score'] * (1 + popularity_boost)

        # Ensure scores don't exceed 1
        all_courses['hybrid_score'] = np.minimum(all_courses['hybrid_score'], 1.0)

    # Exclude already rated courses
    rated_courses = set(user_ratings.keys())
    all_courses = all_courses[~all_courses['Course_ID'].isin(rated_courses)]

    # Sort and get top recommendations
    recommendations = all_courses.sort_values('hybrid_score', ascending=False).head(top_n)

    # Add match score as percentage
    recommendations['Match Score'] = (recommendations['hybrid_score'] * 100).round(1)

    # Add explanation of score components for transparency
    recommendations['Content_Score'] = (recommendations['content_score'] * 100).round(1)
    recommendations['Collaborative_Score'] = (recommendations['collab_score'] * 100).round(1)

    return recommendations


def explain_recommendations(recommendations):
    """
    Provide explanation for why courses were recommended
    """
    explanations = []

    for _, course in recommendations.iterrows():
        explanation = f"Course '{course['Title']}' was recommended because: "

        factors = []
        if course['Content_Score'] > 60:
            factors.append(f"it closely matches your skills and interests ({course['Content_Score']}%)")
        if course['Collaborative_Score'] > 60:
            factors.append(f"it's highly rated by users with similar preferences ({course['Collaborative_Score']}%)")
        if course.get('Rate', 0) > 4.0:
            factors.append(f"it has excellent ratings ({course['Rate']}/5)")
        if course.get('Reviews', 0) > 1000:
            factors.append(f"it's popular among learners ({course['Reviews']} reviews)")

        if factors:
            explanation += ", ".join(factors) + "."
        else:
            explanation += "it's a high-quality course that may interest you."

        explanations.append(explanation)

    return explanations