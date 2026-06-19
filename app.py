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
            background-color: #0a192f;
            color: #ccd6f6;
        }

        /* Input labels */
        label {
            color: #8892b0 !important;
            font-size: 0.9rem !important;
        }

        /* Text inputs — pink */
        .stTextInput > div > div > input {
            background-color: rgba(255, 107, 157, 0.07);
            color: #ccd6f6;
            border: 1px solid #ff6b9d;
            border-radius: 8px;
            padding: 10px 14px;
        }

        .stTextInput > div > div > input::placeholder {
            color: #4a5568;
        }

        .stTextInput > div > div > input:focus {
            border-color: #64ffda;
            box-shadow: 0 0 0 2px rgba(100, 255, 218, 0.15);
        }

        /* Primary button — mint green */
        .stButton > button {
            background-color: transparent;
            color: #64ffda;
            border: 1px solid #64ffda;
            border-radius: 8px;
            padding: 12px 32px;
            font-weight: 700;
            font-size: 1rem;
            width: 100%;
            transition: background 0.2s ease;
        }

        .stButton > button:hover {
            background-color: rgba(100, 255, 218, 0.08);
            color: #64ffda;
        }

        /* Result cards */
        .career-card {
            background-color: #112240;
            border: 1px solid #233554;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
            transition: border-color 0.2s ease;
        }

        .career-card:hover {
            border-color: #ff6b9d;
        }

        .career-card h3 {
            margin: 0 0 6px 0;
            font-size: 1.2rem;
            color: #ccd6f6;
        }

        .career-card p {
            color: #8892b0;
            font-size: 0.88rem;
            margin: 0 0 14px 0;
        }

        .score-bar-bg {
            background-color: #233554;
            border-radius: 999px;
            height: 8px;
            width: 100%;
            margin-bottom: 10px;
        }

        .score-bar-fill {
            background: linear-gradient(90deg, #64ffda, #ff6b9d);
            border-radius: 999px;
            height: 8px;
        }

        .match-label {
            font-size: 0.78rem;
            color: #8892b0;
            margin-bottom: 4px;
        }

        .match-tag {
            display: inline-block;
            background-color: rgba(100, 255, 218, 0.07);
            border: 1px solid rgba(100, 255, 218, 0.2);
            color: #64ffda;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 0.75rem;
            margin: 3px 3px 0 0;
        }

        /* Rank badges */
        .rank-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .rank-1 { background-color: #ff6b9d; color: #0a192f; }
        .rank-2 { background-color: #64ffda; color: #0a192f; }
        .rank-3 { background-color: #ccd6f6; color: #0a192f; }

        /* Divider */
        hr {
            border-color: #233554;
        }

        /* Suggestion chips */
        .suggestion-chip {
            display: inline-block;
            background-color: #112240;
            border: 1px solid #233554;
            color: #8892b0;
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
    input1 = st.text_input("Interest 1", placeholder="e.g. helping people", key="i1")
    input2 = st.text_input("Interest 2", placeholder="e.g. creativity", key="i2")
    input3 = st.text_input("Interest 3", placeholder="e.g. problem solving", key="i3")

with col2:
    input4 = st.text_input("Interest 4 (optional)", placeholder="e.g. working with data", key="i4")
    input5 = st.text_input("Interest 5 (optional)", placeholder="e.g. making a difference", key="i5")
    input6 = st.text_input("Interest 6 (optional)", placeholder="e.g. leadership", key="i6")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CLEAR INDIVIDUAL INPUTS
# ─────────────────────────────────────────────

# Clear buttons per input — only show if field has content
clear_col1, clear_col2 = st.columns(2)

with clear_col1:
    if input1 and st.button("✕ Clear Interest 1", key="c1"):
        st.session_state["i1"] = ""
        st.rerun()
    if input2 and st.button("✕ Clear Interest 2", key="c2"):
        st.session_state["i2"] = ""
        st.rerun()
    if input3 and st.button("✕ Clear Interest 3", key="c3"):
        st.session_state["i3"] = ""
        st.rerun()

with clear_col2:
    if input4 and st.button("✕ Clear Interest 4", key="c4"):
        st.session_state["i4"] = ""
        st.rerun()
    if input5 and st.button("✕ Clear Interest 5", key="c5"):
        st.session_state["i5"] = ""
        st.rerun()
    if input6 and st.button("✕ Clear Interest 6", key="c6"):
        st.session_state["i6"] = ""
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ACTION BUTTONS
# ─────────────────────────────────────────────

btn_col1, btn_col2 = st.columns([3, 1])

with btn_col1:
    run = st.button("Find My Career Paths →")

with btn_col2:
    if st.button("✕ Clear All"):
        for key in ["i1", "i2", "i3", "i4", "i5", "i6"]:
            st.session_state[key] = ""
        st.rerun()

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

        rank_labels  = ["#1 Best Match", "#2 Strong Match", "#3 Good Match"]
        rank_classes = ["rank-1", "rank-2", "rank-3"]

        for i, rec in enumerate(results):
            score_float = rec["score"]
            score_pct   = rec["match_percentage"]
            tags_html   = " ".join([
                f'<span class="match-tag">{t}</span>'
                for t in rec["matched_inputs"]
            ])
            bar_width = min(int(score_float * 100 * 1.8), 100)

            st.markdown(f"""
                <div class="career-card">
                    <div class="rank-badge {rank_classes[i]}">{rank_labels[i]}</div>
                    <h3>{rec['emoji']} {rec['field']}</h3>
                    <p>{rec['description']}</p>
                    <div class="match-label">Match Score: <strong style="color:#64ffda">{score_pct}</strong></div>
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
    "Powered by TF-IDF + Cosine Similarity · DecodeLabs Batch 2026<br>"
    "Built with 🤍 by <span style='color:#ff6b9d;'>Katlego Mathebula</span>"
    "</p>",
    unsafe_allow_html=True
)
