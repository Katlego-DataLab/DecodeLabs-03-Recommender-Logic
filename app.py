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
# COLOUR PALETTE (Brittany Chiang inspired)
# ─────────────────────────────────────────────
# Background       → #0a192f  (deep navy)
# Card / input bg  → #112240  (lighter navy)
# Borders / bars   → #233554  (slate blue)
# Heading (h1)     → #ccd1f6  (soft lavender — "purple" per Brittany palette)
# Body text        → #8892b0  (muted slate)
# Placeholder text → #4a5568  (dark grey)
# Typed input text → #ccd6f6  (soft lavender white)
# Accent mint      → #64ffda
# Accent pink      → #ff6b9d
# Interest box bg  → rgba(255,107,157,0.08)  (pink tint)
# Interest box bdr → #ff6b9d  (pink border)
# Interest text    → #ccd6f6
# ─────────────────────────────────────────────

st.markdown("""
    <style>
        /* ── GLOBAL ── */
        @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;600;700&display=swap');

        .stApp {
            background-color: #0a192f;
            color: #ccd6f6;
            font-family: 'Source Code Pro', 'SF Mono', monospace;
        }

        /* ── HEADER TITLE ── */
        h1, h2, .main-heading {
            color: #ccd1f6 !important;
            font-family: 'Source Code Pro', monospace !important;
        }

        /* ── EYEBROW / SECTION LABELS ── */
        .eyebrow {
            color: #ff6b9d;
            font-size: 0.75rem;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 8px;
            font-family: 'Source Code Pro', monospace;
        }

        /* ── SUBTITLE / BODY TEXT ── */
        p, .stMarkdown p {
            color: #8892b0;
            font-size: 0.9rem;
        }

        /* ── INPUT LABELS ── */
        label {
            color: #8892b0 !important;
            font-size: 0.78rem !important;
            letter-spacing: 1px;
            font-family: 'Source Code Pro', monospace !important;
        }

        /* ── INTEREST INPUT BOXES (pink) ── */
        .stTextInput > div > div > input {
            background-color: rgba(255, 107, 157, 0.07);
            color: #ccd6f6;
            border: 1px solid #ff6b9d;
            border-radius: 6px;
            padding: 10px 14px;
            font-family: 'Source Code Pro', monospace;
            font-size: 0.88rem;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .stTextInput > div > div > input::placeholder {
            color: #4a5568;
        }

        .stTextInput > div > div > input:focus {
            border-color: #64ffda;
            box-shadow: 0 0 0 2px rgba(100, 255, 218, 0.15);
            background-color: rgba(100, 255, 218, 0.04);
            outline: none;
        }

        /* ── PRIMARY BUTTON (mint green) ── */
        .stButton > button {
            background-color: transparent;
            color: #64ffda;
            border: 1px solid #64ffda;
            border-radius: 6px;
            padding: 12px 32px;
            font-weight: 700;
            font-size: 0.88rem;
            font-family: 'Source Code Pro', monospace;
            letter-spacing: 1px;
            width: 100%;
            transition: background 0.2s ease;
        }

        .stButton > button:hover {
            background-color: rgba(100, 255, 218, 0.08);
            color: #64ffda;
        }

        /* ── DIVIDER ── */
        hr {
            border-color: #233554 !important;
        }

        /* ── SUGGESTION CHIPS ── */
        .suggestion-chip {
            display: inline-block;
            background-color: transparent;
            border: 1px solid #233554;
            color: #8892b0;
            border-radius: 4px;
            padding: 4px 12px;
            font-size: 0.75rem;
            margin: 3px;
            font-family: 'Source Code Pro', monospace;
        }

        /* ── RESULT CARDS ── */
        .career-card {
            background-color: #112240;
            border: 1px solid #233554;
            border-radius: 10px;
            padding: 22px 26px;
            margin-bottom: 16px;
            transition: border-color 0.2s ease;
        }

        .career-card:hover {
            border-color: #ff6b9d;
        }

        .career-card h3 {
            margin: 10px 0 4px 0;
            font-size: 1.1rem;
            color: #ccd6f6;
            font-family: 'Source Code Pro', monospace;
            font-weight: 600;
        }

        .career-card p {
            color: #8892b0;
            font-size: 0.82rem;
            margin: 0 0 16px 0;
            line-height: 1.6;
        }

        /* ── SCORE BAR ── */
        .score-bar-bg {
            background-color: #233554;
            border-radius: 999px;
            height: 6px;
            width: 100%;
            margin-bottom: 12px;
        }

        .score-bar-fill {
            background: linear-gradient(90deg, #64ffda, #ff6b9d);
            border-radius: 999px;
            height: 6px;
        }

        /* ── MATCH LABELS ── */
        .match-label {
            font-size: 0.75rem;
            color: #8892b0;
            margin-bottom: 6px;
            font-family: 'Source Code Pro', monospace;
            letter-spacing: 1px;
        }

        /* ── MATCH TAGS ── */
        .match-tag {
            display: inline-block;
            background-color: rgba(100, 255, 218, 0.07);
            border: 1px solid rgba(100, 255, 218, 0.2);
            color: #64ffda;
            border-radius: 3px;
            padding: 3px 10px;
            font-size: 0.72rem;
            margin: 3px 3px 0 0;
            font-family: 'Source Code Pro', monospace;
        }

        /* ── RANK BADGES ── */
        .rank-badge {
            display: inline-block;
            border-radius: 3px;
            padding: 3px 10px;
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: 1px;
            font-family: 'Source Code Pro', monospace;
        }

        .rank-1 { background-color: #ff6b9d; color: #0a192f; }
        .rank-2 { background-color: #64ffda; color: #0a192f; }
        .rank-3 { background-color: #ccd6f6; color: #0a192f; }

        /* ── FOOTER ── */
        .custom-footer {
            text-align: center;
            color: #4a5568;
            font-size: 0.75rem;
            font-family: 'Source Code Pro', monospace;
            margin-top: 40px;
            letter-spacing: 1px;
        }

        .custom-footer .pink  { color: #ff6b9d; }
        .custom-footer .mint  { color: #64ffda; }

        /* ── HIDE STREAMLIT DEFAULTS ── */
        #MainMenu { visibility: hidden; }
        footer    { visibility: hidden; }
        .stDeployButton { display: none; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown('<p class="eyebrow">// career_path_recommender.py</p>', unsafe_allow_html=True)
st.markdown(
    "<h2 style='color:#ccd1f6; font-family:Source Code Pro,monospace; font-weight:700; margin-bottom:4px;'>"
    "Find Your <span style='color:#64ffda;'>Career Path</span></h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:#8892b0; font-size:0.85rem; font-family:Source Code Pro,monospace;'>"
    "Enter at least <strong style='color:#ccd6f6;'>3 interests, values, or things you enjoy</strong> — "
    "the engine matches you to your most aligned careers using TF-IDF + Cosine Similarity.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ─────────────────────────────────────────────
# SUGGESTION CHIPS
# ─────────────────────────────────────────────

st.markdown('<p class="eyebrow">// need inspiration? try words like:</p>', unsafe_allow_html=True)

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

st.markdown('<p class="eyebrow">// your interests & values</p>', unsafe_allow_html=True)

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

run = st.button("→ find my career paths")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if run:
    raw_inputs = [input1, input2, input3, input4, input5, input6]
    user_inputs = [i.strip() for i in raw_inputs if i.strip()]

    if len(user_inputs) < 3:
        st.warning("// Please enter at least 3 interests for an accurate match.")
    else:
        with st.spinner("// running similarity engine..."):
            results = recommend(user_inputs, top_n=3)

        st.markdown("---")
        st.markdown(
            "<h3 style='color:#ccd6f6; font-family:Source Code Pro,monospace; font-weight:600;'>"
            "Your Top Career Matches</h3>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='color:#8892b0; font-size:0.78rem; font-family:Source Code Pro,monospace;'>"
            f"Based on: <span style='color:#64ffda;'>{', '.join(user_inputs)}</span></p>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        rank_labels  = ["#1 best match", "#2 strong match", "#3 good match"]
        rank_classes = ["rank-1", "rank-2", "rank-3"]

        for i, rec in enumerate(results):
            score_float = rec["score"]
            score_pct   = rec["match_percentage"]
            tags_html   = " ".join([
                f'<span class="match-tag">{t}</span>'
                for t in rec["matched_inputs"]
            ])
            bar_width = min(int(score_float * 180), 100)

            st.markdown(f"""
                <div class="career-card">
                    <div class="rank-badge {rank_classes[i]}">{rank_labels[i]}</div>
                    <h3>{rec['emoji']} {rec['field']}</h3>
                    <p>{rec['description']}</p>
                    <div class="match-label">match score: <strong style="color:#64ffda">{score_pct}</strong></div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{bar_width}%"></div>
                    </div>
                    <div class="match-label" style="margin-top:10px">// why this matched</div>
                    {tags_html}
                </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div class='custom-footer'>"
    "Powered by <span class='mint'>TF-IDF + Cosine Similarity</span> "
    "· DecodeLabs Batch 2026<br>"
    "Built with 🤍 by <span class='pink'>Katlego Mathebula</span>"
    "</div>",
    unsafe_allow_html=True
)
