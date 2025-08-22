# src/collaborative.py
import pandas as pd


def collaborative_recommendation(courses: pd.DataFrame, user_ratings: dict, top_n: int = 10):
    """
    Simplified collaborative filtering:
    Recommend top-rated & popular courses the user hasn’t rated.

    courses: pd.DataFrame
    user_ratings: dict {course_title: rating}
    """
    rated_titles = set(user_ratings.keys())

    # Sort by Rate (higher better), then Reviews (popularity)
    ranked = courses.sort_values(
        by=["Rate", "Reviews"], ascending=[False, False]
    )

    # Exclude user-rated courses
    ranked = ranked[~ranked["Title"].isin(rated_titles)]

    return ranked.head(top_n)