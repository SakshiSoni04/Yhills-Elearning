# src/utils.py
from turtle import st

import pandas as pd
import random


def load_courses(file_path):
    """Load courses from CSV file"""
    try:
        courses = pd.read_csv(file_path)
        return courses
    except FileNotFoundError:
        st.error(f"Course data file not found: {file_path}")
        return pd.DataFrame()


def generate_user_profile(user_id, courses_df):
    """Generate a user profile based on available courses"""
    # Extract unique subjects
    subjects = courses_df['Subject'].dropna().unique().tolist()

    # Extract unique levels
    levels = courses_df['Level'].dropna().unique().tolist()

    # Extract skills from the 'Gained Skills' column
    all_skills = []
    for skills in courses_df['Gained Skills'].dropna():
        if isinstance(skills, str):
            all_skills.extend([skill.strip() for skill in skills.split(',')])

    # Get unique skills and select some random ones
    unique_skills = list(set(all_skills))
    selected_skills = random.sample(unique_skills, min(5, len(unique_skills)))

    # Create profile
    profile = {
        'preferred_subjects': random.sample(subjects, min(3, len(subjects))),
        'preferred_levels': random.sample(levels, min(2, len(levels))),
        'skill_interests': selected_skills
    }

    return profile