"""
recommender_logic.py
---------------------
Drop-in replacement for your career recommender logic.

ROOT CAUSE OF THE BUG:
  Using sklearn's TfidfVectorizer or CountVectorizer on tag lists treats
  the entire tag list as a single "document". This means:
    - Multi-word tags like "hands on" get split into ["hands", "on"] → broken
    - TF-IDF down-weights common terms like "food" → wrong scores
    - The vectorizer never sees "baking" as a single unit if spaces split it

THE FIX:
  Use binary cosine similarity on the tags directly (no sklearn vectorizer).
  Each tag is treated as an atomic unit — "hands on" stays as "hands on".
  This is simple, accurate, and fast for this use case.
"""

from math import sqrt


# ── CORE SIMILARITY ──────────────────────────────────────────────────────────

def cosine_similarity_tags(user_tags: list[str], career_tags: list[str]) -> float:
    """
    Compute cosine similarity between user interests and a career's tags.

    Each unique tag across both lists becomes a dimension.
    Vectors are binary: 1 if the tag is present, 0 if not.

    This correctly handles multi-word tags like "hands on" or "making things"
    because we never split them — they're compared as atomic strings.
    """
    # Normalise: lowercase + strip whitespace so "Baking" matches "baking"
    user_set = {t.lower().strip() for t in user_tags}
    career_set = {t.lower().strip() for t in career_tags}

    all_tags = user_set | career_set  # union = all dimensions

    # Build binary vectors
    u_vec = [1 if t in user_set else 0 for t in all_tags]
    c_vec = [1 if t in career_set else 0 for t in all_tags]

    dot_product = sum(a * b for a, b in zip(u_vec, c_vec))
    mag_user = sqrt(sum(a * a for a in u_vec))
    mag_career = sqrt(sum(b * b for b in c_vec))

    if mag_user == 0 or mag_career == 0:
        return 0.0

    return dot_product / (mag_user * mag_career)


def get_matching_tags(user_tags: list[str], career_tags: list[str]) -> list[str]:
    """Return the tags that appear in BOTH the user's interests and a career."""
    user_set = {t.lower().strip() for t in user_tags}
    career_set = {t.lower().strip() for t in career_tags}
    return sorted(user_set & career_set)


# ── MAIN RECOMMENDER ─────────────────────────────────────────────────────────

def recommend_careers(
    user_interests: list[str],
    career_profiles: list[dict],
    top_n: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    """
    Rank career profiles by cosine similarity to the user's interests.

    Args:
        user_interests:   List of interest/value strings the user entered.
        career_profiles:  The CAREER_PROFILES list from career_profiles.py.
        top_n:            How many results to return (default 5).
        min_score:        Minimum similarity score to include (0.0 = no filter).

    Returns:
        List of dicts, sorted by score descending, each containing:
          - field, emoji, description  (from the profile)
          - score       (float 0–1)
          - score_pct   (score as a percentage string, e.g. "50.0%")
          - matched_tags (list of tags that caused the match)
          - rank_label  ("#1 Best Match", "#2 Strong Match", etc.)
    """
    if not user_interests:
        return []

    results = []
    for profile in career_profiles:
        score = cosine_similarity_tags(user_interests, profile["tags"])
        if score > min_score:
            results.append({
                "field":        profile["field"],
                "emoji":        profile["emoji"],
                "description":  profile["description"],
                "score":        score,
                "score_pct":    f"{score * 100:.1f}%",
                "matched_tags": get_matching_tags(user_interests, profile["tags"]),
            })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_n]

    # Add rank labels
    rank_labels = [
        "#1 Best Match", "#2 Strong Match", "#3 Good Match",
        "#4 Possible Match", "#5 Worth Exploring"
    ]
    for i, result in enumerate(results):
        result["rank"] = i + 1
        result["rank_label"] = rank_labels[i] if i < len(rank_labels) else f"#{i+1} Match"

    return results


# ── QUICK TEST ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Paste your CAREER_PROFILES here for testing, or import them
    from career_profiles_extended import CAREER_PROFILES  # adjust import as needed

    test_cases = [
        ["food", "hospitality", "baking", "hands on", "creativity"],
        ["animals", "biology", "helping", "care", "nature"],
        ["sports", "coaching", "fitness", "teamwork", "motivation"],
        ["mathematics", "data", "patterns", "logic", "statistics"],
        ["mining", "engineering", "outdoors", "geology", "safety"],
    ]

    for interests in test_cases:
        print(f"\n{'='*60}")
        print(f"USER INTERESTS: {interests}")
        print("="*60)
        results = recommend_careers(interests, CAREER_PROFILES, top_n=3)
        for r in results:
            print(f"  {r['rank_label']}: {r['emoji']} {r['field']}")
            print(f"    Score: {r['score_pct']}  |  Matched: {r['matched_tags']}")
