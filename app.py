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
    page_title="Find Your Path",
    page_icon="🧭",
    layout="centered"
)

# ─────────────────────────────────────────────
# SESSION STATE — interest chips
# ─────────────────────────────────────────────

if "interests" not in st.session_state:
    st.session_state.interests = []

if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# ─────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        /* ── BACKGROUND ── */
        .stApp {
            background-color: #0a192f;
            color: #ccd6f6;
        }

        /* ── MAIN HEADING: "Find Your" ── */
        .main-heading {
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            font-size: 4.2rem;
            color: #ccd6f6;
            line-height: 1.0;
            margin-bottom: 0px;
        }

        /* ── SLOGAN: "Career Path" — teal, same large weight ── */
        .slogan {
            font-family: 'Inter', sans-serif;
            font-weight: 800;
            font-size: 4.2rem;
            color: #64ffda;
            line-height: 1.0;
            margin-top: 0px;
            margin-bottom: 24px;
        }

        /* ── SUBTITLE ── */
        .subtitle {
            color: #8892b0;
            font-size: 0.9rem;
        }

        /* ── INPUT LABELS ── */
        label {
            color: #8892b0 !important;
            font-size: 0.9rem !important;
        }

        /* ── TEXT INPUT (for adding interests) ── */
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

        /* ── BUTTONS ── */
        .stButton > button {
            background-color: transparent;
            color: #64ffda;
            border: 1px solid #64ffda;
            border-radius: 8px;
            padding: 10px 28px;
            font-weight: 700;
            font-size: 0.95rem;
            width: 100%;
            transition: background 0.2s ease;
        }

        .stButton > button:hover {
            background-color: rgba(100, 255, 218, 0.08);
            color: #64ffda;
        }

        /* ── INTEREST CHIPS ── */
        .chips-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 14px;
            background-color: #112240;
            border: 1px solid #233554;
            border-radius: 10px;
            min-height: 54px;
            margin-bottom: 12px;
            align-items: center;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background-color: rgba(168, 85, 247, 0.12);
            border: 1px solid #a855f7;
            color: #ccd6f6;
            border-radius: 999px;
            padding: 5px 14px;
            font-size: 0.82rem;
            font-weight: 500;
        }

        .chip-x {
            color: #a855f7;
            font-size: 0.75rem;
            font-weight: 700;
            cursor: pointer;
            margin-left: 2px;
        }

        .chips-empty {
            color: #4a5568;
            font-size: 0.82rem;
            font-style: italic;
        }

        /* ── RESULT CARDS ── */
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

        /* ── RANK BADGES ── */
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
        .rank-3 { background-color: #a855f7; color: #ffffff; }

        /* ── SUGGESTION CHIPS ROW ── */
        .suggestion-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 16px;
        }

        .suggestion-chip-btn {
            display: inline-block;
            background-color: #112240;
            border: 1px solid #233554;
            color: #8892b0;
            border-radius: 999px;
            padding: 6px 16px;
            font-size: 0.82rem;
            cursor: pointer;
            transition: border-color 0.2s, color 0.2s;
            white-space: nowrap;
        }

        .suggestion-chip-btn:hover {
            border-color: #64ffda;
            color: #64ffda;
        }

        /* ── DIVIDER ── */
        hr { border-color: #233554; }

        /* ── HIDE STREAMLIT DEFAULTS ── */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stDeployButton { display: none; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown('<p class="main-heading">Find Your</p>', unsafe_allow_html=True)
st.markdown('<p class="slogan">Career Path</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Add your interests and values below — '
    'the engine matches you to your most aligned careers using TF-IDF + Cosine Similarity.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ─────────────────────────────────────────────
# SUGGESTION CHIPS
# ─────────────────────────────────────────────

st.markdown("**Need inspiration? Click to add:**")

suggestions = [
    "helping people", "creativity", "problem solving", "working with data",
    "nature", "writing", "leadership", "making a difference",
    "technology", "art", "justice", "building things", "research",
    "music", "community", "mathematics", "storytelling", "independence"
]

# Render suggestion chips in horizontal wrapping rows (6 per row)
row_size = 6
for row_start in range(0, len(suggestions), row_size):
    row_items = suggestions[row_start:row_start + row_size]
    cols = st.columns(len(row_items))
    for idx, sug in enumerate(row_items):
        with cols[idx]:
            if st.button(sug, key=f"sug_{row_start + idx}"):
                if sug not in st.session_state.interests and len(st.session_state.interests) < 6:
                    st.session_state.interests.append(sug)
                    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INTEREST CHIPS DISPLAY
# ─────────────────────────────────────────────

st.markdown("**Your Selected Interests**")

# Show existing chips with individual X buttons
if st.session_state.interests:
    # Render chips as HTML display
    chips_html = "".join([
        f'<span class="chip">{interest} <span class="chip-x">✕</span></span>'
        for interest in st.session_state.interests
    ])
    st.markdown(f'<div class="chips-container">{chips_html}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="chips-container"><span class="chips-empty">No interests added yet — type below or click a suggestion above</span></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ADD INTEREST INPUT
# ─────────────────────────────────────────────

st.markdown("**Add an Interest or Value**")

add_col, btn_col = st.columns([4, 1])

with add_col:
    new_interest = st.text_input(
        "Type an interest",
        placeholder="e.g. helping people, creativity, problem solving...",
        label_visibility="collapsed",
        key="new_interest_input"
    )

with btn_col:
    if st.button("＋ Add"):
        cleaned = new_interest.strip()
        if cleaned and cleaned not in st.session_state.interests:
            if len(st.session_state.interests) < 6:
                st.session_state.interests.append(cleaned)
                st.rerun()
            else:
                st.warning("Maximum 6 interests reached.")
        elif cleaned in st.session_state.interests:
            st.warning("Already added!")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ACTION BUTTONS
# ─────────────────────────────────────────────

run_col, clear_col = st.columns([3, 1])

with run_col:
    run = st.button("Find My Career Paths →")

with clear_col:
    if st.button("✕ Clear All"):
        st.session_state.interests = []
        st.rerun()

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if run:
    user_inputs = st.session_state.interests

    if len(user_inputs) < 3:
        st.warning("Please add at least 3 interests for an accurate match.")
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
    "<p style='color:#8b949e; font-size:0.82rem; text-align:center; margin-top: 40px;'>"
    "Powered by TF-IDF + Cosine Similarity · DecodeLabs Batch 2026<br>"
    "Built with 🤍 by <span style='color:#ff6b9d;'>Katlego Mathebula</span>"
    "</p>",
    unsafe_allow_html=True
)
