"""
Career Path Recommender, Streamlit App
─────────────────────────────────────────────
A content-based filtering system that maps user interests
and values to the most aligned career fields using
TF-IDF vectorization and Cosine Similarity.

Built by: Katlego Mathebula
Internship: DecodeLabs 2026 | Project 3
"""

import streamlit as st
from recommender import recommend

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Career Path Recommender",
    page_icon="🧭",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────

st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #0f0f0f;
            color: #f0f0f0;
        }

        /* Input labels */
        label {
            color: #c9d1d9 !important;
            font-size: 0.9rem !important;
        }

        /* Text inputs */
        .stTextInput > div > div > input {
            background-color: #1c1c1e;
            color: #f0f0f0;
            border: 1px solid #2d2d2f;
            border-radius: 8px;
            padding: 10px 14px;
        }

        /* Button */
        .stButton > button {
            background-color: #d4af37;
            color: #0f0f0f;
            border: none;
            border-radius: 8px;
            padding: 12px 32px;
            font-weight: 700;
            font-size: 1rem;
            width: 100%;
            transition: opacity 0.2s ease;
        }

        .stButton > button:hover {
            opacity: 0.85;
            color: #0f0f0f;
        }

        /* Result cards */
        .career-card {
            background-color: #1c1c1e;
            border: 1px solid #2d2d2f;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
        }

        .career-card h3 {
            margin: 0 0 6px 0;
            font-size: 1.2rem;
            color: #f0f0f0;
        }

        .career-card p {
            color: #8b949e;
            font-size: 0.88rem;
            margin: 0 0 14px 0;
        }

        .score-bar-bg {
            background-color: #2d2d2f;
            border-radius: 999px;
            height: 8px;
            width: 100%;
            margin-bottom: 10px;
        }

        .score-bar-fill {
            background-color: #d4af37;
            border-radius: 999px;
            height: 8px;
        }

        .match-label {
            font-size: 0.78rem;
            color: #8b949e;
            margin-bottom: 4px;
        }

        .match-tag {
            display: inline-block;
            background-color: #2d2d2f;
            color: #d4af37;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 0.75rem;
            margin: 3px 3px 0 0;
        }

        /* Rank badge */
        .rank-badge {
            display: inline-block;
            background-color: #d4af37;
            color: #0f0f0f;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        /* Divider */
        hr {
            border-color: #2d2d2f;
        }

        /* Suggestion chips */
        .suggestion-chip {
            display: inline-block;
            background-color: #1c1c1e;
            border: 1px solid #2d2d2f;
            color: #8b949e;
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 0.78rem;
            margin: 3px;
        }

        /* Hide streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("## 🧭 Career Path Recommender")
st.markdown(
    "Enter at least **3 interests, values, or things you enjoy** — "
    "and the engine will match you to your most aligned career paths using AI similarity logic."
)

st.markdown("---")

# ─────────────────────────────────────────────
# SUGGESTION CHIPS (UX HINT)
# ─────────────────────────────────────────────

st.markdown("**Need inspiration? Try words like:**")
suggestions = [
    "helping people", "creativity", "problem solving", "working with data",
    "nature", "writing", "leadership", "making a difference",
    "technology", "art", "justice", "building things", "research",
    "music", "community", "mathematics", "storytelling", "independence"
]
chips_html = " ".join([f'<span class="suggestion-chip">{s}</span>' for s in suggestions])
st.markdown(chips_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# USER INPUT FORM
# ─────────────────────────────────────────────

st.markdown("**Your Interests & Values**")

col1, col2 = st.columns(2)

with col1:
    input1 = st.text_input("Interest 1", placeholder="e.g. helping people")
    input2 = st.text_input("Interest 2", placeholder="e.g. creativity")
    input3 = st.text_input("Interest 3", placeholder="e.g. problem solving")

with col2:
    input4 = st.text_input("Interest 4 (optional)", placeholder="e.g. working with data")
    input5 = st.text_input("Interest 5 (optional)", placeholder="e.g. making a difference")
    input6 = st.text_input("Interest 6 (optional)", placeholder="e.g. leadership")

st.markdown("<br>", unsafe_allow_html=True)

run = st.button("Find My Career Paths →")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if run:
    raw_inputs = [input1, input2, input3, input4, input5, input6]
    user_inputs = [i.strip() for i in raw_inputs if i.strip()]

    if len(user_inputs) < 3:
        st.warning("Please enter at least 3 interests for an accurate match.")
    else:
        with st.spinner("Running similarity engine..."):
            results = recommend(user_inputs, top_n=3)

        st.markdown("---")
        st.markdown("### Your Top Career Matches")
        st.markdown(f"*Based on: {', '.join(user_inputs)}*")
        st.markdown("<br>", unsafe_allow_html=True)

        rank_labels = ["#1 Best Match", "#2 Strong Match", "#3 Good Match"]

        for i, rec in enumerate(results):
            score_float = rec["score"]
            score_pct = rec["match_percentage"]
            tags_html = " ".join([
                f'<span class="match-tag">{t}</span>'
                for t in rec["matched_inputs"]
            ])
            bar_width = min(int(score_float * 100 * 1.8), 100)

            st.markdown(f"""
                <div class="career-card">
                    <div class="rank-badge">{rank_labels[i]}</div>
                    <h3>{rec['emoji']} {rec['field']}</h3>
                    <p>{rec['description']}</p>
                    <div class="match-label">Match Score: <strong style="color:#d4af37">{score_pct}</strong></div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{bar_width}%"></div>
                    </div>
                    <div class="match-label" style="margin-top:10px">Why this matched:</div>
                    {tags_html}
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<p style='color:#8b949e; font-size:0.82rem; text-align:center;'>"
            "Powered by TF-IDF + Cosine Similarity · DecodeLabs Batch 2026 · Built by Katlego Mathebula"
            "</p>",
            unsafe_allow_html=True
        )
