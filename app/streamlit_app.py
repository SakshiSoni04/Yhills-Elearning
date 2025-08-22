# app/streamlit_app.py
import streamlit as st
import pandas as pd
import sys, os, random

# Add parent dir (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_courses
from src.hybrid import hybrid_recommendation

# ----------------- Config -----------------
st.set_page_config(page_title="E-Learning Recommendation System", page_icon="🎓", layout="wide")

# ----------------- Sidebar Theme Switch -----------------
theme = st.sidebar.radio("Theme", ["🌞", "🌙"], index=0, horizontal=False)

# ----------------- Theme Colors -----------------
if theme == "🌞":
    bg_color, text_color, card_bg = "#f9fafb", "#111827", "#ffffff"
else:
    bg_color, text_color, card_bg = "#111827", "#f9fafb", "#1f2937"

# ----------------- Custom CSS -----------------
st.markdown(f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Segoe UI', sans-serif;
    }}
    h1 {{
        text-align: center;
        color: #2563EB;
    }}
    .course-container {{
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-start;
    }}
    .course-card {{
        background: {card_bg};
        padding: 16px;
        border-radius: 12px;
        margin: 10px;
        flex: 1 1 calc(33% - 20px);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .course-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
    }}
    .course-title {{
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 6px;
        color: {text_color};
    }}
    .course-meta {{
        font-size: 0.9em;
        color: #6B7280;
        margin-bottom: 6px;
    }}
    .course-skill {{
        font-size: 0.85em;
        color: {text_color};
    }}
    .score-bar {{
        height: 8px;
        border-radius: 6px;
        background: #E5E7EB;
        margin-top: 10px;
        overflow: hidden;
    }}
    .score-fill {{
        height: 8px;
        border-radius: 6px;
        background: linear-gradient(90deg, #2563EB, #4F46E5);
    }}
    .score-text {{
        font-size: 0.85em;
        color: #2563EB;
        margin-top: 4px;
    }}
    </style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.title("🎓 Personalized E-Learning Recommendation System")

# ----------------- Load Dataset -----------------
courses = load_courses("data/Coursera.csv")

# ----------------- Sub-header -----------------
st.markdown("<p style='text-align:center;color:gray'>Discover the best courses tailored to your skills & interests</p>", unsafe_allow_html=True)

# ----------------- Search + Filters -----------------
with st.container():
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_text = st.text_input("🔍 Search courses or skills", "", placeholder="Python, Data Science, Marketing")
    with col2:
        show_filters = st.checkbox("⚙️ Show Filters")
    with col3:
        top_n = st.number_input("📌 No. of courses", min_value=5, max_value=20, value=10)

# Default filter values
subject_filter, level_filter, institution_filter = [], [], []
min_rate = 3.5

if show_filters:
    with st.expander("Filter Options"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            subjects = sorted(courses["Subject"].dropna().unique().tolist())
            subject_filter = st.multiselect("📚 Subject", subjects, default=subjects)
        with col2:
            levels = sorted(courses["Level"].dropna().unique().tolist())
            level_filter = st.multiselect("🎯 Level", levels, default=levels)
        with col3:
            institutions = sorted(courses["Institution"].dropna().unique().tolist())
            institution_filter = st.multiselect("🏫 Institution", institutions, default=institutions)
        with col4:
            min_rate = st.slider("⭐ Min Rating", 0.0, 5.0, min_rate, 0.1)

# ----------------- User Skills Input -----------------
user_skills = st.text_input("📝 Enter your interests/skills (comma separated):", "machine learning, python, finance")

# ----------------- Filter Dataset -----------------
filtered = courses.copy()
if search_text:
    mask = (
        filtered["Title"].str.contains(search_text, case=False, na=False) |
        filtered["Gained Skills"].str.contains(search_text, case=False, na=False)
    )
    filtered = filtered[mask]
if subject_filter:
    filtered = filtered[filtered["Subject"].isin(subject_filter)]
if level_filter:
    filtered = filtered[filtered["Level"].isin(level_filter)]
if institution_filter:
    filtered = filtered[filtered["Institution"].isin(institution_filter)]
filtered = filtered[filtered["Rate"] >= min_rate]

# ----------------- Recommendations -----------------
if st.button("🚀 Get Recommendations"):
    recs = hybrid_recommendation(filtered, user_skills, {}, top_n=top_n)

    if recs.empty:
        st.warning("No recommendations found. Try changing your filters or skills.")
    else:
        st.markdown("<div class='course-container'>", unsafe_allow_html=True)

        # Placeholder images for subjects
        subject_images = {
            "Business": "https://img.icons8.com/color/96/briefcase.png",
            "Data Science": "https://img.icons8.com/color/96/artificial-intelligence.png",
            "Computer Science": "https://img.icons8.com/color/96/laptop.png",
            "Health": "https://img.icons8.com/color/96/heart-health.png",
            "Arts": "https://img.icons8.com/color/96/paint-palette.png",
        }

        for _, row in recs.iterrows():
            match = int(float(row.get("Match Score", 0)) * 100) if "Match Score" in row else random.randint(50,90)
            subject_img = subject_images.get(row["Subject"], "https://img.icons8.com/color/96/book.png")

            st.markdown(f"""
            <div class="course-card">
                <img src="{subject_img}" width="60">
                <div class="course-title">{row['Title']}</div>
                <div class="course-meta">🏫 {row['Institution']} | 🎯 {row['Level']} | ⏳ {row['Duration']}</div>
                <div class="course-meta">⭐ {row['Rate']} ({row['Reviews']} reviews)</div>
                <div class="course-skill"><b>Skills:</b> {row['Gained Skills']}</div>
                <div class="score-bar"><div class="score-fill" style="width: {match}%;"></div></div>
                <div class="score-text">Match Score: {match}%</div>
            </div>
            """, unsafe_allow_html=True)

            # More Info Expander
            with st.expander("ℹ️ More Info"):
                st.write("**Subject:**", row['Subject'])
                st.write("**Learning Product:**", row['Learning Product'])
                st.write("**Reviews:**", row['Reviews'])
                st.write("**Skills Covered:**", row['Gained Skills'])

        st.markdown("</div>", unsafe_allow_html=True)