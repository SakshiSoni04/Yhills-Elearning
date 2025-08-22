# src/hybrid.py
import pandas as pd
from src.content_based import content_based_recommendation
from src.collaborative import collaborative_recommendation

def hybrid_recommendation(courses: pd.DataFrame, user_skills: str, user_ratings: dict, top_n: int = 10):
    """
    Hybrid = Content-based + Collaborative
    We take half recommendations from each.
    """
    n_content = top_n // 2
    n_collab = top_n - n_content

    content_recs = content_based_recommendation(courses, user_skills, top_n=n_content)
    collab_recs = collaborative_recommendation(courses, user_ratings, top_n=n_collab)

    # Combine and drop duplicates
    final = pd.concat([content_recs, collab_recs]).drop_duplicates(subset=["Title"])
    return final.head(top_n)