# hybrid.py
import pandas as pd
from src.content_based import content_based_recommendation
from src.collaborative import collaborative_recommendation
import numpy as np


def hybrid_recommendation(courses: pd.DataFrame, user_skills: str, user_ratings: dict,
                          user_id: int = None, top_n: int = 10, alpha: float = 0.7):
    """
    Enhanced hybrid recommendation with weighted combination
    """
    # Make a copy of the courses DataFrame to avoid modifying the original
    courses_copy = courses.copy()

    content_recs = content_based_recommendation(courses_copy, user_skills, user_id, top_n=top_n * 2)
    collab_recs = collaborative_recommendation(courses_copy, user_ratings, top_n=top_n * 2)

    all_courses = courses_copy.copy()
    all_courses['content_score'] = 0
    all_courses['collab_score'] = 0

    # Map content-based scores
    content_scores = dict(zip(content_recs['Course_ID'], content_recs['similarity']))
    all_courses['content_score'] = all_courses['Course_ID'].map(content_scores).fillna(0)

    # Map collaborative filtering scores
    if not collab_recs.empty and 'Predicted_Rating' in collab_recs.columns:
        collab_scores = dict(zip(collab_recs['Course_ID'], collab_recs['Predicted_Rating']))
        all_courses['collab_score'] = all_courses['Course_ID'].map(collab_scores).fillna(0)
    else:
        all_courses['collab_score'] = 0

    # Normalize scores
    content_max = all_courses['content_score'].max()
    collab_max = all_courses['collab_score'].max()

    if content_max > 0:
        all_courses['content_score'] = all_courses['content_score'] / content_max

    if collab_max > 0:
        all_courses['collab_score'] = all_courses['collab_score'] / collab_max

    all_courses['hybrid_score'] = (
            alpha * all_courses['content_score'] +
            (1 - alpha) * all_courses['collab_score']
    )

    # Exclude already rated courses
    rated_courses = set(user_ratings.keys())
    all_courses = all_courses[~all_courses['Course_ID'].isin(rated_courses)]

    recommendations = all_courses.sort_values('hybrid_score', ascending=False).head(top_n)
    recommendations['Match Score'] = recommendations['hybrid_score']

    return recommendations