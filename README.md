# Career Path Recommender
### Content-Based Filtering Engine | DecodeLabs Batch 2026 | Project 3

---

## Overview

A recommendation system that maps a user's personal interests and values to the most aligned career fields using TF-IDF vectorization and Cosine Similarity, built from scratch without ML libraries.

The system solves a real problem: people struggle to identify career paths that genuinely match who they are, not just what they studied. By treating career exploration as a pattern-matching problem, this engine delivers objective, ranked guidance through pure algorithmic logic.

---

## Technical Architecture

The engine follows a strict **Input → Process → Output (IPO)** pipeline:

```
User Interests (text)
        ↓
  [1] Ingestion       — clean and tokenize user input
        ↓
  [2] TF-IDF Vectors  — transform text into weighted numerical arrays
        ↓
  [3] Cosine Scoring  — measure angular similarity between user and each career profile
        ↓
  [4] Sort + Filter   — rank descending, return Top-3
        ↓
Ranked Career Matches + Match % + Explanation
```

---

## Why These Algorithms?

**TF-IDF over binary matching:**
Simple tag counting treats "helping people" and "software" as equally important. TF-IDF weights terms by how specific they are rare, descriptive tags carry more influence than generic ones, producing more nuanced recommendations.

**Cosine Similarity over Euclidean Distance:**
Euclidean distance is sensitive to vector magnitude. A user who enters 6 interests would unfairly outscore a user who enters 3 even if their profile direction is identical. Cosine similarity measures the angle between vectors, making it magnitude-invariant and far more accurate for text data.

---

## Key Features

- Accepts 3–6 free-text interest/value inputs
- TF-IDF + Cosine Similarity computed from scratch (no sklearn)
- 15 career field profiles across diverse domains
- Interpretability layer: shows which inputs drove each recommendation
- Cold start handler: fallback to trending paths when input is empty
- Clean Streamlit UI with match score progress bars

---

## Project Structure

```
career_recommender/
│
├── app.py                  # Streamlit interface
├── recommender.py          # Core TF-IDF + Cosine engine
├── career_profiles.py      # Curated career field dataset
└── README.md
```

---

## How to Run

```bash
# Install dependency
pip install streamlit

# Launch app
streamlit run app.py
```

---

## Sample Output

**Input:** `helping people, creativity, making a difference, storytelling, community`

| Rank | Career Field | Match Score | Why Matched |
|------|-------------|-------------|-------------|
| #1 | 📚 Education & Teaching | 44.9% | helping people, creativity, making a difference, storytelling |
| #2 | 🤝 Social Work & Community Development | 33.6% | helping people, making a difference, community |
| #3 | 🧠 Psychology & Mental Health | 29.8% | helping people, making a difference |

---

## Built with 🤍 by Katlego Mathebula  
DecodeLabs Internship — AI engineering   
[GitHub](https://github.com/Katlego-DataLab) · [Portfolio](https://katlego-datalab.github.io/Website-updated-/) · [LinkedIn](https://www.linkedin.com/in/katlego-mathebula)
