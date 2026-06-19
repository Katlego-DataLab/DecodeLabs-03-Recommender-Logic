"""
Career Path Recommender - Core Engine
 ─────────────────────────────────────────────
Uses TF-IDF vectorization and Cosine Similarity to match
user interests/values to career field profiles.

Pipeline:
    1. Ingestion   — collect and clean user inputs
    2. Scoring     — TF-IDF vectorize + cosine similarity
    3. Sorting     — rank by similarity score descending
    4. Filtering   — return Top-N results with explanations
"""

import math
from collections import Counter
from career_profiles import CAREER_PROFILES


# ─────────────────────────────────────────────
# STEP 1: TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(text: str) -> list[str]:
    """Lowercase and tokenize a string into individual words."""
    return text.lower().strip().split()


def preprocess_tags(tags: list[str]) -> list[str]:
    """Flatten and tokenize a list of tag phrases into individual words."""
    tokens = []
    for tag in tags:
        tokens.extend(preprocess(tag))
    return tokens


def build_user_tokens(user_inputs: list[str]) -> list[str]:
    """
    Convert raw user inputs into a token list.
    Expands multi-word phrases so they map correctly
    to the shared vocabulary space.
    """
    tokens = []
    for inp in user_inputs:
        tokens.extend(preprocess(inp))
    return tokens


# ─────────────────────────────────────────────
# STEP 2: BUILD VOCABULARY
# ─────────────────────────────────────────────

def build_vocabulary(profiles: list[dict]) -> list[str]:
    """
    Build a shared vocabulary from all career profile tags.
    This is the vector space every profile and user maps into.
    """
    vocab = set()
    for profile in profiles:
        tokens = preprocess_tags(profile["tags"])
        vocab.update(tokens)
    return sorted(list(vocab))


# ─────────────────────────────────────────────
# STEP 3: TF-IDF VECTORIZATION
# ─────────────────────────────────────────────

def compute_tf(tokens: list[str], vocab: list[str]) -> dict:
    """
    Term Frequency: how often each vocab word appears
    in this document (career profile or user input).
    TF(t) = count(t) / total_tokens
    """
    total = len(tokens) if tokens else 1
    count = Counter(tokens)
    return {word: count.get(word, 0) / total for word in vocab}


def compute_idf(profiles: list[dict], vocab: list[str]) -> dict:
    """
    Inverse Document Frequency: penalizes words that appear
    in too many career profiles (generic), rewards rare ones (specific).
    IDF(t) = log(total_docs / docs_containing_t)
    """
    total_docs = len(profiles)
    idf = {}
    for word in vocab:
        docs_with_word = sum(
            1 for p in profiles
            if word in preprocess_tags(p["tags"])
        )
        # Smoothing: add 1 to avoid division by zero
        idf[word] = math.log(total_docs / (1 + docs_with_word)) + 1
    return idf


def compute_tfidf_vector(tokens: list[str], vocab: list[str], idf: dict) -> list[float]:
    """
    TF-IDF Vector: combines TF and IDF into a weighted numerical array.
    This is what gets compared using cosine similarity.
    """
    tf = compute_tf(tokens, vocab)
    return [tf[word] * idf[word] for word in vocab]


# ─────────────────────────────────────────────
# STEP 4: COSINE SIMILARITY
# ─────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Measures the angle between two vectors.
    Invariant to magnitude — focuses purely on orientation.
    Score of 1.0 = perfect alignment, 0.0 = no overlap.

    cos(θ) = (A · B) / (||A|| × ||B||)
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # Cold start, no overlap possible

    return dot_product / (magnitude_a * magnitude_b)


# ─────────────────────────────────────────────
# STEP 5: EXPLAINABILITY, WHY THIS MATCHED
# ─────────────────────────────────────────────

def get_matched_tags(user_inputs: list[str], profile_tags: list[str]) -> list[str]:
    """
    Returns which user input phrases semantically overlap
    with the career profile tags, the 'why this matched' layer.
    """
    user_words = set(build_user_tokens(user_inputs))
    profile_words = set(preprocess_tags(profile_tags))
    matched_words = user_words.intersection(profile_words)

    # Map matched words back to original user input phrases
    matched_phrases = []
    for inp in user_inputs:
        inp_words = set(preprocess(inp))
        if inp_words.intersection(matched_words):
            matched_phrases.append(inp)

    return matched_phrases if matched_phrases else ["general interest alignment"]


# ─────────────────────────────────────────────
# STEP 6: COLD START HANDLER
# ─────────────────────────────────────────────

def handle_cold_start(profiles: list[dict], top_n: int = 3) -> list[dict]:
    """
    When user provides no meaningful input (zero vector),
    fall back to top trending/popular career fields.
    """
    popular = ["Technology & Software Engineering", "Healthcare & Medicine", "Business & Entrepreneurship"]
    results = []
    for profile in profiles:
        if profile["field"] in popular:
            results.append({
                "field": profile["field"],
                "emoji": profile["emoji"],
                "description": profile["description"],
                "score": 0.0,
                "match_percentage": "N/A",
                "matched_inputs": ["Tell us your interests for a personalised match!"]
            })
    return results[:top_n]


# ─────────────────────────────────────────────
# MAIN RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────

def recommend(user_inputs: list[str], top_n: int = 3) -> list[dict]:
    """
    Full recommendation pipeline.

    Args:
        user_inputs : list of user interest/value strings (min 3 recommended)
        top_n       : number of recommendations to return

    Returns:
        List of dicts with field, score, match_percentage, matched_inputs
    """
    # Clean inputs
    user_inputs = [inp.strip() for inp in user_inputs if inp.strip()]

    if not user_inputs:
        return handle_cold_start(CAREER_PROFILES, top_n)

    # Build shared vocabulary and IDF weights
    vocab = build_vocabulary(CAREER_PROFILES)
    idf = compute_idf(CAREER_PROFILES, vocab)

    # Vectorize user profile
    user_tokens = build_user_tokens(user_inputs)
    user_vector = compute_tfidf_vector(user_tokens, vocab, idf)

    # Check for zero vector (cold start)
    if all(v == 0 for v in user_vector):
        return handle_cold_start(CAREER_PROFILES, top_n)

    # Score every career profile
    results = []
    for profile in CAREER_PROFILES:
        profile_tokens = preprocess_tags(profile["tags"])
        profile_vector = compute_tfidf_vector(profile_tokens, vocab, idf)

        score = cosine_similarity(user_vector, profile_vector)
        matched = get_matched_tags(user_inputs, profile["tags"])

        results.append({
            "field": profile["field"],
            "emoji": profile["emoji"],
            "description": profile["description"],
            "score": round(score, 4),
            "match_percentage": f"{round(score * 100, 1)}%",
            "matched_inputs": matched
        })

    # Sort descending by score, return Top-N
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# ─────────────────────────────────────────────
# QUICK CLI TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_inputs = ["helping people", "creativity", "making a difference", "storytelling"]
    print("\n🔍 Testing recommender with inputs:", test_inputs)
    print("─" * 50)

    recommendations = recommend(test_inputs)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n#{i} {rec['emoji']} {rec['field']}")
        print(f"   Match Score    : {rec['match_percentage']}")
        print(f"   Why Matched    : {', '.join(rec['matched_inputs'])}")
        print(f"   Description    : {rec['description']}")
