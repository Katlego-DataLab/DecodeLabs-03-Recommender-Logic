# 🧭 Career Path Recommender, Find Your Career Path
### Recommender Logic, DecodeLabs Batch 2026 | Project 03

---

![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![No ML Libraries](https://img.shields.io/badge/Built%20From%20Scratch-No%20sklearn-64ffda?style=for-the-badge&logo=databricks&logoColor=black)
![Algorithm](https://img.shields.io/badge/Algorithm-TF--IDF%20%2B%20Cosine%20Similarity-a855f7?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen?style=for-the-badge)
![DecodeLabs](https://img.shields.io/badge/DecodeLabs-2026-ff6b9d?style=for-the-badge)

## Try the LIVE 
[![Live App](https://img.shields.io/badge/▶%20LIVE%20APP-Open%20in%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=1a0000)](https://decodelabs-03-recommender-logic-project03mathebula03katlego.streamlit.app/)

---

## Overview

A recommendation system that maps a user's personal interests and values to the most aligned career fields using **TF-IDF vectorization** and **Cosine Similarity** built entirely from scratch without ML libraries.

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
- Quick-add suggestion chips for faster input
- Removable interest chips with duplicate detection
- Clean dark-themed Streamlit UI with match score progress bars

---

## Project Structure

```
career_recommender/
│
├── app.py                  # Streamlit interface
├── recommender.py          # Core TF-IDF + Cosine engine
├── recommender_logic.py    # Binary cosine similarity (drop-in alternative)
├── career_profiles.py      # Curated career field dataset
└── README.md
```

---

## How to Run Locally

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
| #1 Best Match | 📚 Education & Teaching | 44.9% | helping people, creativity, making a difference, storytelling |
| #2 Strong Match | 🤝 Social Work & Community Development | 33.6% | helping people, making a difference, community |
| #3 Good Match | 🧠 Psychology & Mental Health | 29.8% | helping people, making a difference |

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Vectorization | Custom TF-IDF (pure Python) |
| Similarity | Custom Cosine Similarity (pure Python) |
| Data | Hand-curated career profiles |
| Deployment | Streamlit Community Cloud |

---

## Built with 🤍 by Katlego Mathebula
DecodeLabs Internship 2026, AI Engineering

[![GitHub](https://img.shields.io/badge/GitHub-Katlego--DataLab-181717?style=flat-square&logo=github)](https://github.com/Katlego-DataLab)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-ff6b9d?style=flat-square&logo=vercel&logoColor=white)](https://katlego-datalab.github.io/Website-updated-/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/katlego-mathebula)
