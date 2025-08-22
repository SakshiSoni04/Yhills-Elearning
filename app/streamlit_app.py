# app/streamlit_app.py
import streamlit as st
import pandas as pd
import sys, os

# Add parent dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.models import DatabaseManager
from src.utils import load_courses, generate_user_profile
from src.hybrid import hybrid_recommendation

# ----------------- Config -----------------
st.set_page_config(page_title="E-Learning Recommendation System", page_icon="🎓", layout="wide")

# ----------------- MySQL Configuration -----------------
MYSQL_CONFIG = {
    "host": "localhost",
    "database": "elearning_db",
    "user": "root",
    "password": "Sahil@2006",
    "port": 3306
}
db = DatabaseManager(**MYSQL_CONFIG)

# ----------------- Sidebar Theme Switch -----------------
theme = st.sidebar.radio("Theme", ["🌞 Light", "🌙 Dark"], index=1)
if theme.startswith("🌞"):
    bg_color, text_color, card_bg, info_bg = "#f9fafb", "#111827", "#ffffff", "#f0f4f8"
else:
    bg_color, text_color, card_bg, info_bg = "#111827", "#f9fafb", "#1f2937", "#2d3748"

# ----------------- Custom CSS -----------------
st.markdown(f"""
<style>
body {{ background-color: {bg_color}; color: {text_color}; font-family: 'Segoe UI', sans-serif; }}
.main-header {{ text-align: center; color: #2563EB; font-size: 2rem; margin-bottom: 0.5rem; }}
.sub-header {{ text-align: center; color: gray; font-size: 1rem; margin-bottom: 1.5rem; }}
.course-card {{
    background-color: {card_bg};
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}}
.skill-pill {{
    display: inline-block;
    background-color: #2563EB;
    color: white;
    padding: 4px 10px;
    border-radius: 16px;
    margin: 3px;
    font-size: 0.75rem;
    cursor: pointer;
    border: none;
    text-align: center;
    min-width: 60px;
}}
.skill-pill:hover {{
    background-color: #1E40AF;
    transform: scale(1.05);
}}
.skills-container {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 6px;
    margin-bottom: 1rem;
}}
.compact-button {{
    font-size: 0.8rem;
    padding: 3px 8px;
    margin: 2px;
    height: auto;
    line-height: 1.2;
    border-radius: 12px;
}}
.stButton > button {{
    border-radius: 12px;
}}
.stTextInput > div > div > input {{
    border-radius: 12px;
}}
.stNumberInput > div > div > input {{
    border-radius: 12px;
}}
.stCheckbox > label {{
    border-radius: 12px;
}}
.stSelectbox > div > div {{
    border-radius: 12px;
}}
.stMultiSelect > div > div {{
    border-radius: 12px;
}}
.stSlider > div {{
    border-radius: 12px;
}}
.stExpander {{
    border-radius: 12px;
    border: 1px solid #e1e5e9;
}}
.course-info-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-top: 10px;
}}
.course-info-item {{
    background-color: {info_bg};
    padding: 8px;
    border-radius: 8px;
    font-size: 0.9rem;
    color: {text_color};
}}
.skill-button-row {{
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}}
.skill-button {{
    background-color: #2563EB;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s ease;
}}
.skill-button:hover {{
    background-color: #1E40AF;
    transform: scale(1.05);
}}
</style>
""", unsafe_allow_html=True)

# ----------------- Header -----------------
st.markdown("<h1 class='main-header'>🎓 E-Learning Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Discover courses tailored to your skills & interests</p>", unsafe_allow_html=True)


# ----------------- Authentication -----------------
def login_form():
    with st.sidebar:
        st.header("Authentication")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                user_id = db.verify_user(username, password)
                if user_id > 0:
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    profile = db.get_user_profile(user_id)
                    if not profile:
                        courses_df = load_courses("data/Coursera.csv")
                        if 'Course_ID' not in courses_df.columns:
                            courses_df['Course_ID'] = courses_df['Title'].str.slice(0, 20) + courses_df[
                                'Institution'].str.slice(0, 10)
                            courses_df['Course_ID'] = courses_df['Course_ID'].str.replace(' ', '_')
                        profile_data = generate_user_profile(user_id, courses_df)
                        db.save_user_profile(user_id, profile_data)
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        with tab2:
            new_user = st.text_input("Username", key="signup_user")
            new_pass = st.text_input("Password", type="password", key="signup_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
            if st.button("Create Account"):
                if new_pass != confirm_pass:
                    st.error("Passwords don't match")
                else:
                    user_id = db.create_user(new_user, new_pass)
                    if user_id > 0:
                        st.success("Account created! Please login.")
                    else:
                        st.error("Username already exists")


# ----------------- Load Dataset -----------------
@st.cache_data
def load_courses_cached(path):
    df = load_courses(path)
    if 'Course_ID' not in df.columns:
        df['Course_ID'] = df['Title'].str.slice(0, 20) + df['Institution'].str.slice(0, 10)
        df['Course_ID'] = df['Course_ID'].str.replace(' ', '_')
    return df


courses = load_courses_cached("data/Coursera.csv")


# ----------------- Extract Top Skills -----------------
@st.cache_data
def get_top_skills(courses_df, top_n=10):
    # Extract all skills from the Gained Skills column
    all_skills = []
    for skills_str in courses_df['Gained Skills'].dropna():
        skills = [skill.strip() for skill in skills_str.split(',')]
        all_skills.extend(skills)

    # Count frequency of each skill
    skill_counts = pd.Series(all_skills).value_counts()
    return skill_counts.head(top_n).index.tolist()


top_skills = get_top_skills(courses)


# ----------------- Main App -----------------
def main_app():
    user_profile = db.get_user_profile(st.session_state.user_id)
    default_skills = ""
    if user_profile and user_profile.get('skill_interests'):
        default_skills = ", ".join(user_profile['skill_interests'])

    # ----------------- Top Skills Section -----------------
    st.subheader("🔥 Popular Skills")

    # Initialize session state for search text if not exists
    if 'search_text' not in st.session_state:
        st.session_state.search_text = ""

    # Create two rows of skill buttons
    rows = [top_skills[:5], top_skills[5:10]]

    for row_skills in rows:
        cols = st.columns(len(row_skills))
        for col_idx, skill in enumerate(row_skills):
            with cols[col_idx]:
                if st.button(
                        skill,
                        key=f"skill_{skill}",
                        help=f"Add '{skill}' to search",
                        use_container_width=True
                ):
                    # Add skill to search when clicked
                    current_skills = [s.strip() for s in st.session_state.search_text.split(',') if s.strip()]
                    if skill not in current_skills:
                        if st.session_state.search_text:
                            st.session_state.search_text += f", {skill}"
                        else:
                            st.session_state.search_text = skill
                    st.rerun()

    st.markdown("---")

    # ----------------- Search + Filters -----------------
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            search_text = st.text_input(
                "🔍 Search courses or skills",
                value=st.session_state.search_text,
                placeholder="Python, Data Science, Marketing",
                key="search_input"
            )
            st.session_state.search_text = search_text
        with col2:
            show_filters = st.checkbox("⚙️ Show Filters")
        with col3:
            top_n = st.number_input("📌 No. of courses", min_value=5, max_value=20, value=10)

    subject_filter, level_filter, institution_filter = [], [], []
    min_rate = 3.5

    # Initialize filter state if not exists
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False

    if show_filters:
        with st.expander("Filter Options", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                subjects = ["All"] + sorted(courses["Subject"].dropna().unique().tolist())
                subject_selection = st.multiselect("📚 Subject", subjects, default=["All"])
                if "All" in subject_selection:
                    subject_filter = []
                else:
                    subject_filter = subject_selection
            with col2:
                levels = ["All"] + sorted(courses["Level"].dropna().unique().tolist())
                level_selection = st.multiselect("🎯 Level", levels, default=["All"])
                if "All" in level_selection:
                    level_filter = []
                else:
                    level_filter = level_selection
            with col3:
                institutions = ["All"] + sorted(courses["Institution"].dropna().unique().tolist())
                institution_selection = st.multiselect("🏫 Institution", institutions, default=["All"])
                if "All" in institution_selection:
                    institution_filter = []
                else:
                    institution_filter = institution_selection
            with col4:
                min_rate = st.slider("⭐ Min Rating", 0.0, 5.0, min_rate, 0.1)

            # Apply filters button
            if st.button("Apply Filters"):
                st.session_state.filters_applied = True
                st.session_state.subject_filter = subject_filter
                st.session_state.level_filter = level_filter
                st.session_state.institution_filter = institution_filter
                st.session_state.min_rate = min_rate
                st.rerun()

            # Clear filters button
            if st.button("Clear Filters"):
                st.session_state.filters_applied = False
                if 'subject_filter' in st.session_state:
                    del st.session_state.subject_filter
                if 'level_filter' in st.session_state:
                    del st.session_state.level_filter
                if 'institution_filter' in st.session_state:
                    del st.session_state.institution_filter
                st.rerun()

    # Use stored filter values if filters are applied
    if st.session_state.filters_applied:
        subject_filter = st.session_state.get('subject_filter', [])
        level_filter = st.session_state.get('level_filter', [])
        institution_filter = st.session_state.get('institution_filter', [])
        min_rate = st.session_state.get('min_rate', 3.5)

    # ----------------- User Interests Input -----------------
    user_skills_input = st.text_input(
        "📝 Enter your interests/skills (comma separated):",
        value=default_skills or "machine learning, python, finance"
    )

    # ----------------- Filter Courses -----------------
    filtered = courses.copy()

    # Apply search filter
    if search_text:
        search_terms = [term.strip() for term in search_text.split(',')]
        mask = pd.Series(False, index=filtered.index)
        for term in search_terms:
            if term:
                term_mask = (
                        filtered["Title"].str.contains(term, case=False, na=False) |
                        filtered["Gained Skills"].str.contains(term, case=False, na=False)
                )
                mask = mask | term_mask
        filtered = filtered[mask]

    # Apply other filters only if they're not empty
    if subject_filter:
        filtered = filtered[filtered["Subject"].isin(subject_filter)]
    if level_filter:
        filtered = filtered[filtered["Level"].isin(level_filter)]
    if institution_filter:
        filtered = filtered[filtered["Institution"].isin(institution_filter)]

    filtered = filtered[filtered["Rate"] >= min_rate]

    # ----------------- Recommendations -----------------
    recs = pd.DataFrame()
    if st.button("🚀 Get Recommendations"):
        recs = hybrid_recommendation(filtered, user_skills_input, {}, st.session_state.user_id, top_n=top_n)

    if recs.empty:
        if search_text or st.session_state.filters_applied:
            display_courses = filtered
            st.info("Showing filtered courses.")
        else:
            display_courses = pd.DataFrame()  # Show nothing by default
    else:
        display_courses = recs

    # ----------------- Horizontal Course Display -----------------
    if not display_courses.empty:
        subject_images = {
            "Business": "https://img.icons8.com/color/96/briefcase.png",
            "Data Science": "https://img.icons8.com/color/96/artificial-intelligence.png",
            "Computer Science": "https://img.icons8.com/color/96/laptop.png",
            "Health": "https://img.icons8.com/color/96/heart-health.png",
            "Arts": "https://img.icons8.com/color/96/paint-palette.png",
        }

        for idx, row in display_courses.iterrows():
            # Fix match score calculation - ensure it's always a float between 0-1
            match_score = row.get("Match Score", 0)

            # Handle different formats of match score
            if isinstance(match_score, str):
                if '%' in match_score:
                    match_score = float(match_score.replace('%', '')) / 100
                else:
                    try:
                        match_score = float(match_score)
                        # If it's > 1, assume it's a percentage and convert to decimal
                        if match_score > 1:
                            match_score = match_score / 100
                    except ValueError:
                        match_score = 0
            elif isinstance(match_score, (int, float)):
                # If it's > 1, assume it's a percentage and convert to decimal
                if match_score > 1:
                    match_score = match_score / 100

            # Ensure match_score is between 0 and 1
            match_score = max(0, min(1, match_score))
            match_percentage = int(match_score * 100)

            subject_img = subject_images.get(row.get("Subject", "General"), "https://img.icons8.com/color/96/book.png")

            st.markdown(f"<div class='course-card'>", unsafe_allow_html=True)

            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(subject_img, width=80)
            with col2:
                st.markdown(f"### {row.get('Title', 'No Title')}")
                st.markdown(
                    f"🏫 **{row.get('Institution', 'N/A')}** | 🎯 {row.get('Level', 'N/A')} | ⏳ {row.get('Duration', 'N/A')}")
                st.markdown(f"⭐ **{row.get('Rate', 'N/A')}** ({row.get('Reviews', 'N/A')} reviews)")

                # Skills with limited display
                skills = row.get('Gained Skills', 'No skills listed')
                if len(str(skills)) > 100:
                    skills = str(skills)[:100] + "..."
                st.markdown(f"**Skills:** {skills}")

                # Fixed match score display
                st.progress(match_score)
                st.markdown(f"**Match Score:** {match_percentage}%")

                # More Info button with expander - simplified to show only required fields
                with st.expander("📖 Course Details"):
                    # Create a grid layout for course details
                    st.markdown("""
                    <div class="course-info-grid">
                        <div class="course-info-item"><strong>Subject:</strong> {}</div>
                        <div class="course-info-item"><strong>Title:</strong> {}</div>
                        <div class="course-info-item"><strong>Institution:</strong> {}</div>
                        <div class="course-info-item"><strong>Learning Product:</strong> {}</div>
                        <div class="course-info-item"><strong>Level:</strong> {}</div>
                        <div class="course-info-item"><strong>Duration:</strong> {}</div>
                        <div class="course-info-item"><strong>Gained Skills:</strong> {}</div>
                        <div class="course-info-item"><strong>Rate:</strong> {}</div>
                        <div class="course-info-item"><strong>Reviews:</strong> {}</div>
                    </div>
                    """.format(
                        row.get('Subject', 'N/A'),
                        row.get('Title', 'N/A'),
                        row.get('Institution', 'N/A'),
                        row.get('Learning Product', 'N/A'),
                        row.get('Level', 'N/A'),
                        row.get('Duration', 'N/A'),
                        row.get('Gained Skills', 'N/A'),
                        row.get('Rate', 'N/A'),
                        row.get('Reviews', 'N/A')
                    ), unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ----------------- App Flow -----------------
if 'user_id' not in st.session_state:
    login_form()
else:
    st.sidebar.write(f"Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    main_app()