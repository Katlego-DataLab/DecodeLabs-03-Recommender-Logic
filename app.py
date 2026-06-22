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
# SESSION STATE
# ─────────────────────────────────────────────

if "interests" not in st.session_state:
    st.session_state.interests = []

if "new_interest_input" not in st.session_state:
    st.session_state.new_interest_input = ""

# transient UI messages (shown once, then cleared)
if "duplicate_msg" not in st.session_state:
    st.session_state.duplicate_msg = False

if "max_msg" not in st.session_state:
    st.session_state.max_msg = False

MAX_INTERESTS = 6

# ─────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&family=Poppins:wght@400;600;700;800;900&display=swap');

        /* ── BACKGROUND ── */
        .stApp {
            background-color: #0a192f;
            color: #ccd6f6;
        }

        /* ── HIDE STREAMLIT TOOLBAR (Share / Star / Edit / GitHub) ── */
        header[data-testid="stHeader"] {
            visibility: hidden;
            height: 0;
        }
        [data-testid="stToolbar"] {
            visibility: hidden;
            display: none;
        }
        #MainMenu { visibility: hidden; }
        .stDeployButton { display: none; }

        /* ── MAIN HEADING: "Find Your" ── */
        .main-heading {
            font-family: 'Poppins', 'Inter', sans-serif;
            font-weight: 900;
            font-size: 3rem;
            color: #ccd6f6;
            line-height: 0.9;
            margin-bottom: 0px;
            letter-spacing: -0.02em;
        }

        /* ── SECOND LINE: "CAREER PATH" — purple gradient + glow ── */
        .career-path-line {
            font-family: 'Poppins', 'Inter', sans-serif;
            font-weight: 900;
            font-size: 4rem;
            line-height: 0.9;
            margin-top: 4px;
            margin-bottom: 18px;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #A855F7, #C084FC);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: none;
            filter: drop-shadow(0 0 20px rgba(168, 85, 247, 0.35));
        }

        /* ── SLOGAN: "Find Your Future." ── */
        .slogan {
            font-family: 'Poppins', 'Inter', sans-serif;
            font-weight: 600;
            font-size: 1rem;
            color: #a855f7;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-top: 0px;
            margin-bottom: 12px;
        }

        /* ── VALUE PROP ── */
        .subtitle {
            color: #8892b0;
            font-family: 'Inter', sans-serif;
            font-size: 1.2rem;
            font-weight: 400;
            margin-top: 0;
            margin-bottom: 8px;
        }

        /* ── RESPONSIVE TITLE SIZING ── */
        @media (max-width: 768px) {
            .main-heading { font-size: 2rem; }
            .career-path-line { font-size: 2.8rem; }
        }
        @media (max-width: 480px) {
            .main-heading { font-size: 1.6rem; }
            .career-path-line { font-size: 2.2rem; }
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

        /* ── BUTTONS (default) ── */
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

        /* ── SELECTED INTEREST CHIPS BOX (background/border, holds the horizontal row) ── */
        .st-key-chips_box {
            background-color: #112240;
            border: 1px solid #233554;
            border-radius: 10px;
            padding: 14px;
            min-height: 26px;
        }

        .st-key-chips_row .stButton {
            width: auto !important;
        }

        .st-key-chips_row .stButton > button {
            background-color: rgba(168, 85, 247, 0.12);
            border: 1px solid #a855f7;
            color: #ccd6f6;
            border-radius: 999px;
            padding: 5px 14px;
            font-size: 0.82rem;
            font-weight: 500;
            width: auto;
            white-space: nowrap;
        }

        .st-key-chips_row .stButton > button:hover {
            background-color: rgba(255, 107, 157, 0.15);
            border-color: #ff6b9d;
            color: #ff6b9d;
        }

        .chips-empty {
            color: #4a5568;
            font-size: 0.82rem;
            font-style: italic;
            padding: 14px;
            background-color: #112240;
            border: 1px solid #233554;
            border-radius: 10px;
            min-height: 26px;
            margin-bottom: 12px;
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
        .st-key-suggestions_row .stButton > button {
            border-radius: 12px !important;
            padding: 7px 16px !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            white-space: nowrap !important;
            width: auto !important;
            min-width: 0 !important;
            border-color: #233554 !important;
            color: #8892b0 !important;
        }
        .st-key-suggestions_row .stButton > button:hover {
            border-color: #64ffda !important;
            color: #64ffda !important;
        }

        /* ── DIVIDER ── */
        hr { border-color: #233554; }

        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────

def add_interest(text=None):
    """Add an interest, clear the input field, and surface any messages."""
    cleaned = (text if text is not None else st.session_state.new_interest_input).strip()

    st.session_state.duplicate_msg = False
    st.session_state.max_msg = False

    if not cleaned:
        st.session_state.new_interest_input = ""
        return

    existing_lower = [i.lower() for i in st.session_state.interests]
    if cleaned.lower() in existing_lower:
        st.session_state.duplicate_msg = True
    elif len(st.session_state.interests) >= MAX_INTERESTS:
        st.session_state.max_msg = True
    else:
        st.session_state.interests.append(cleaned)

    # Always clear the typed text after an add attempt
    st.session_state.new_interest_input = ""


def add_interest_from_text_input():
    add_interest()


def add_suggestion(sug):
    def _cb():
        add_interest(sug)
    return _cb


def remove_interest(index):
    def _cb():
        if 0 <= index < len(st.session_state.interests):
            st.session_state.interests.pop(index)
    return _cb


def clear_all():
    st.session_state.interests = []
    st.session_state.new_interest_input = ""
    st.session_state.duplicate_msg = False
    st.session_state.max_msg = False


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown('<p class="main-heading">Find Your</p>', unsafe_allow_html=True)
st.markdown('<p class="career-path-line">CAREER PATH</p>', unsafe_allow_html=True)
st.markdown('<p class="slogan">Find Your Future.</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Turn your interests into career opportunities.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ─────────────────────────────────────────────
# SUGGESTION CHIPS — 10 examples pulled from across the dataset's industries
# ─────────────────────────────────────────────

st.markdown("**Need inspiration? Click to add:**")

suggestions = [
    "helping people",      # Healthcare, Education, Social Work...
    "creativity",           # Creative Arts, Tech, Marketing...
    "problem solving",      # Tech, Engineering, Finance...
    "working with data",    # Data Science, Finance, Statistics...
    "teaching",              # Education
    "sports",                # Sports, Fitness & Coaching
    "nature",                 # Environmental Science, Agriculture
    "leadership",             # Business, Education, Sports
    "storytelling",           # Media, Creative Arts, Education
    "building things",        # Engineering, Tech, Trades
]

with st.container(key="suggestions_row", horizontal=True, gap="xsmall"):
    for sug in suggestions:
        st.button(sug, key=f"sug_{sug}", on_click=add_suggestion(sug))

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INTEREST CHIPS DISPLAY — each chip is now a real, removable button
# ─────────────────────────────────────────────

st.markdown("**Your Selected Interests**")

if st.session_state.interests:
    with st.container(key="chips_box"):
        with st.container(key="chips_row", horizontal=True, gap="xsmall"):
            for i, interest in enumerate(st.session_state.interests):
                st.button(
                    f"{interest}  ✕",
                    key=f"remove_{i}_{interest}",
                    on_click=remove_interest(i)
                )
else:
    st.markdown(
        '<div class="chips-empty">No interests added yet — type below or click a suggestion above</div>',
        unsafe_allow_html=True
    )

if st.session_state.duplicate_msg:
    st.caption("⚠️ Already added.")
if st.session_state.max_msg:
    st.caption(f"⚠️ Maximum {MAX_INTERESTS} interests reached.")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ADD INTEREST INPUT
# ─────────────────────────────────────────────

st.markdown("**Add an Interest or Value**")

add_col, btn_col = st.columns([4, 1])

with add_col:
    st.text_input(
        "Type an interest",
        placeholder="e.g. helping people, creativity, problem solving...",
        label_visibility="collapsed",
        key="new_interest_input",
        on_change=add_interest_from_text_input
    )

with btn_col:
    st.button("＋ Add", on_click=add_interest_from_text_input)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ACTION BUTTONS
# ─────────────────────────────────────────────

run_col, clear_col = st.columns([3, 1])

with run_col:
    run = st.button("Find My Career Paths →")

with clear_col:
    st.button("✕ Clear All", on_click=clear_all)

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

        rank_labels = ["#1 Best Match", "#2 Strong Match", "#3 Good Match"]
        rank_classes = ["rank-1", "rank-2", "rank-3"]

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
