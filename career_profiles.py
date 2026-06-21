"""
Career Path Recommender — Matching Engine
─────────────────────────────────────────────
Content-based filtering: each career field is represented as a "profile"
of descriptive tags (interests, values, work styles, traits, themes).
A user's selected interests are vectorised with the same vocabulary and
compared against every profile using TF-IDF + Cosine Similarity.

Built by: Katlego Mathebula
Internship: DecodeLabs 2026 | Project 3
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# CAREER PROFILES DATASET
# ─────────────────────────────────────────────

CAREER_PROFILES = [

    # ── ORIGINAL PROFILES ────────────────────────────────────────────────────

    {
        "field": "Healthcare & Medicine",
        "emoji": "🏥",
        "description": "Diagnosing, treating, and caring for people's physical and mental wellbeing.",
        "tags": [
            "helping people", "science", "biology", "problem solving", "empathy",
            "human body", "research", "teamwork", "precision", "critical thinking",
            "care", "service", "medicine", "health", "compassion", "detail oriented",
            "decision making", "people skills", "saving lives", "community"
        ]
    },
    {
        "field": "Education & Teaching",
        "emoji": "📚",
        "description": "Shaping minds, transferring knowledge, and empowering the next generation.",
        "tags": [
            "teaching", "helping people", "communication", "patience", "mentoring",
            "knowledge sharing", "creativity", "empathy", "community", "leadership",
            "learning", "child development", "language", "storytelling", "guidance",
            "making a difference", "people skills", "inspiration", "collaboration", "social impact"
        ]
    },
    {
        "field": "Technology & Software Engineering",
        "emoji": "💻",
        "description": "Building digital products, solving complex problems through code and systems thinking.",
        "tags": [
            "problem solving", "logic", "building things", "technology", "innovation",
            "computers", "mathematics", "critical thinking", "creativity", "detail oriented",
            "automation", "systems thinking", "coding", "data", "engineering",
            "independence", "continuous learning", "precision", "efficiency", "future"
        ]
    },
    {
        "field": "Data Science & AI",
        "emoji": "🤖",
        "description": "Extracting insights from data and building intelligent systems that learn.",
        "tags": [
            "data", "mathematics", "problem solving", "patterns", "research",
            "technology", "statistics", "critical thinking", "innovation", "logic",
            "machine learning", "artificial intelligence", "curiosity", "analysis",
            "computers", "continuous learning", "automation", "future", "science",
            "precision", "data storytelling"
        ]
    },
    {
        "field": "Creative Arts & Design",
        "emoji": "🎨",
        "description": "Expressing ideas visually, aesthetically, and emotionally through creative work.",
        "tags": [
            "creativity", "art", "visual thinking", "self expression", "aesthetics",
            "storytelling", "design", "innovation", "independence", "imagination",
            "beauty", "culture", "communication", "emotion", "hands on",
            "making things", "freedom", "originality", "inspiration", "craftsmanship"
        ]
    },
    {
        "field": "Law & Justice",
        "emoji": "⚖️",
        "description": "Upholding rights, navigating legal systems, and advocating for fairness.",
        "tags": [
            "justice", "critical thinking", "arguing", "research", "writing",
            "ethics", "fairness", "problem solving", "helping people", "society",
            "language", "persuasion", "detail oriented", "logic", "debate",
            "human rights", "advocacy", "making a difference", "precision", "leadership"
        ]
    },
    {
        "field": "Business & Entrepreneurship",
        "emoji": "📈",
        "description": "Creating ventures, managing organisations, and driving economic value.",
        "tags": [
            "leadership", "innovation", "risk taking", "problem solving", "strategy",
            "communication", "independence", "ambition", "creativity", "building things",
            "money", "teamwork", "persuasion", "networking", "decision making",
            "freedom", "growth", "opportunity", "responsibility", "future"
        ]
    },
    {
        "field": "Finance & Accounting",
        "emoji": "💰",
        "description": "Managing money, analysing financial systems, and enabling economic decisions.",
        "tags": [
            "mathematics", "money", "detail oriented", "analysis", "precision",
            "logic", "problem solving", "systems thinking", "data", "research",
            "responsibility", "critical thinking", "patterns", "planning", "stability",
            "economics", "trust", "accuracy", "strategy", "numbers"
        ]
    },
    {
        "field": "Social Work & Community Development",
        "emoji": "🤝",
        "description": "Supporting vulnerable communities and driving social change at the grassroots.",
        "tags": [
            "helping people", "empathy", "community", "social impact", "making a difference",
            "compassion", "advocacy", "human rights", "service", "collaboration",
            "listening", "people skills", "fairness", "resilience", "patience",
            "poverty", "mental health", "children", "justice", "society"
        ]
    },
    {
        "field": "Environmental Science & Sustainability",
        "emoji": "🌍",
        "description": "Protecting the planet, studying ecosystems, and building a sustainable future.",
        "tags": [
            "nature", "science", "research", "problem solving", "sustainability",
            "environment", "animals", "biology", "future", "making a difference",
            "systems thinking", "data", "critical thinking", "outdoors", "conservation",
            "climate", "responsibility", "community", "innovation", "curiosity"
        ]
    },
    {
        "field": "Media, Journalism & Communications",
        "emoji": "📰",
        "description": "Informing, influencing, and connecting people through storytelling and media.",
        "tags": [
            "writing", "storytelling", "communication", "creativity", "language",
            "curiosity", "people skills", "research", "society", "culture",
            "persuasion", "independence", "truth", "social impact", "technology",
            "interviewing", "investigation", "current events", "self expression", "influence"
        ]
    },
    {
        "field": "Psychology & Mental Health",
        "emoji": "🧠",
        "description": "Understanding human behaviour, emotions, and supporting mental wellbeing.",
        "tags": [
            "empathy", "helping people", "human behaviour", "listening", "science",
            "research", "mental health", "compassion", "critical thinking", "curiosity",
            "patterns", "analysis", "people skills", "care", "emotion",
            "problem solving", "society", "wellbeing", "making a difference", "patience"
        ]
    },
    {
        "field": "Engineering & Architecture",
        "emoji": "🏗️",
        "description": "Designing and building physical structures, systems, and infrastructure.",
        "tags": [
            "building things", "mathematics", "problem solving", "precision", "design",
            "engineering", "creativity", "systems thinking", "hands on", "critical thinking",
            "innovation", "detail oriented", "planning", "teamwork", "logic",
            "making things", "technology", "responsibility", "craftsmanship", "future"
        ]
    },
    {
        "field": "Agriculture & Food Science",
        "emoji": "🌱",
        "description": "Feeding the world through farming innovation, food technology, and rural development.",
        "tags": [
            "nature", "science", "hands on", "outdoors", "sustainability",
            "biology", "community", "problem solving", "research", "making a difference",
            "food", "environment", "innovation", "responsibility", "animals",
            "economics", "planning", "systems thinking", "rural", "health"
        ]
    },
    {
        "field": "Performing Arts & Entertainment",
        "emoji": "🎭",
        "description": "Moving audiences through music, theatre, dance, film, and live performance.",
        "tags": [
            "creativity", "self expression", "performance", "storytelling", "emotion",
            "art", "culture", "communication", "imagination", "freedom",
            "music", "entertainment", "inspiration", "people skills", "originality",
            "beauty", "passion", "collaboration", "confidence", "making things"
        ]
    },

    # ── SPORTS & FITNESS ──────────────────────────────────────────────────────

    {
        "field": "Sports, Fitness & Coaching",
        "emoji": "🏆",
        "description": "Training athletes, promoting physical wellness, and competing or coaching at every level.",
        "tags": [
            "sports", "fitness", "competition", "teamwork", "leadership",
            "performance", "motivation", "discipline", "health", "people skills",
            "coaching", "strategy", "resilience", "physical activity", "outdoors",
            "community", "goal setting", "communication", "passion", "making a difference"
        ]
    },
    {
        "field": "Sports Science & Physiotherapy",
        "emoji": "🦴",
        "description": "Applying science to enhance athletic performance, prevent injury, and rehabilitate the body.",
        "tags": [
            "sports", "science", "human body", "health", "research",
            "biology", "helping people", "precision", "problem solving", "fitness",
            "rehabilitation", "analysis", "care", "teamwork", "performance",
            "critical thinking", "detail oriented", "medicine", "hands on", "wellbeing"
        ]
    },
    {
        "field": "Sports Commentary & Broadcasting",
        "emoji": "🎙️",
        "description": "Bringing sports to life through live commentary, analysis, and broadcasting for audiences worldwide.",
        "tags": [
            "sports", "communication", "storytelling", "performance", "language",
            "media", "entertainment", "passion", "current events", "research",
            "people skills", "confidence", "culture", "interviewing", "live events",
            "creativity", "influence", "knowledge sharing", "teamwork", "public speaking"
        ]
    },

    # ── FOOD & HOSPITALITY ────────────────────────────────────────────────────

    {
        "field": "Culinary Arts & Baking",
        "emoji": "🍽️",
        "description": "Crafting exceptional food and baked goods through creativity, technique, and passion for flavour.",
        "tags": [
            "food", "creativity", "hands on", "craftsmanship", "making things",
            "art", "science", "precision", "culture", "passion",
            "innovation", "aesthetics", "detail oriented", "self expression", "originality",
            "sensory", "entrepreneurship", "hospitality", "baking", "community"
        ]
    },
    {
        "field": "Hospitality & Food Business",
        "emoji": "🍴",
        "description": "Running restaurants, hotels, and food ventures where guest experience and business acumen meet.",
        "tags": [
            "food", "leadership", "people skills", "entrepreneurship", "hospitality",
            "business", "communication", "teamwork", "creativity", "customer service",
            "management", "problem solving", "money", "community", "culture",
            "strategy", "ambition", "building things", "networking", "responsibility"
        ]
    },

    # ── ANIMALS & NATURE ──────────────────────────────────────────────────────

    {
        "field": "Veterinary & Animal Science",
        "emoji": "🐾",
        "description": "Caring for animals' health, welfare, and behaviour across pets, livestock, and wildlife.",
        "tags": [
            "animals", "science", "biology", "helping", "care",
            "empathy", "medicine", "health", "hands on", "nature",
            "research", "problem solving", "compassion", "detail oriented", "critical thinking",
            "outdoors", "responsibility", "precision", "curiosity", "making a difference"
        ]
    },

    # ── BUSINESS & FINANCE ────────────────────────────────────────────────────

    {
        "field": "Finance, Investment & Wealth Management",
        "emoji": "📊",
        "description": "Growing capital, managing risk, and building financial strategies for individuals and institutions.",
        "tags": [
            "money", "mathematics", "strategy", "analysis", "risk taking",
            "ambition", "patterns", "critical thinking", "data", "economics",
            "problem solving", "logic", "leadership", "research", "precision",
            "numbers", "future", "decision making", "responsibility", "innovation"
        ]
    },
    {
        "field": "Marketing & Brand Strategy",
        "emoji": "📣",
        "description": "Building brands, understanding consumers, and crafting campaigns that move people to act.",
        "tags": [
            "creativity", "communication", "persuasion", "storytelling", "strategy",
            "business", "people skills", "innovation", "data", "culture",
            "social impact", "psychology", "research", "ambition", "digital",
            "branding", "self expression", "language", "networking", "influence"
        ]
    },
    {
        "field": "Supply Chain, Logistics & Operations",
        "emoji": "🚚",
        "description": "Keeping goods, services, and systems flowing efficiently across organisations and borders.",
        "tags": [
            "problem solving", "systems thinking", "planning", "efficiency", "logistics",
            "mathematics", "strategy", "detail oriented", "teamwork", "technology",
            "responsibility", "precision", "data", "economics", "critical thinking",
            "process", "engineering", "global", "decision making", "operations"
        ]
    },
    {
        "field": "Project Management",
        "emoji": "📋",
        "description": "Planning, executing, and delivering projects on time and on budget across any industry.",
        "tags": [
            "planning", "leadership", "organisation", "teamwork", "problem solving",
            "communication", "strategy", "responsibility", "detail oriented", "decision making",
            "efficiency", "systems thinking", "goal setting", "collaboration", "risk management",
            "process", "deadline driven", "critical thinking", "accountability", "coordination"
        ]
    },
    {
        "field": "Human Resources & Talent Acquisition",
        "emoji": "🧑‍💼",
        "description": "Finding, developing, and retaining great people while shaping positive workplace cultures.",
        "tags": [
            "people skills", "communication", "empathy", "listening", "organisation",
            "leadership", "teamwork", "fairness", "problem solving", "strategy",
            "interviewing", "talent", "culture", "collaboration", "negotiation",
            "making a difference", "mentoring", "planning", "responsibility", "human behaviour"
        ]
    },

    # ── CREATIVE & MEDIA ──────────────────────────────────────────────────────

    {
        "field": "Film & Television Production",
        "emoji": "🎬",
        "description": "Producing, directing, and creating compelling content for the screen — from concept to final cut.",
        "tags": [
            "creativity", "storytelling", "leadership", "collaboration", "art",
            "communication", "technology", "detail oriented", "performance", "imagination",
            "culture", "entertainment", "planning", "self expression", "teamwork",
            "innovation", "problem solving", "music", "visual thinking", "passion"
        ]
    },

    # ── TRADES & SKILLED LABOUR ───────────────────────────────────────────────

    {
        "field": "Farming & Agricultural Production",
        "emoji": "🚜",
        "description": "Working the land to grow food and raise livestock, balancing tradition with modern technique.",
        "tags": [
            "nature", "outdoors", "hands on", "animals", "responsibility",
            "community", "sustainability", "food", "rural", "independence",
            "biology", "environment", "resilience", "planning", "making things",
            "physical activity", "seasons", "patience", "economics", "hard work"
        ]
    },
    {
        "field": "Plumbing & Building Services",
        "emoji": "🔧",
        "description": "Installing and maintaining water, gas, and heating systems that keep homes and buildings running.",
        "tags": [
            "hands on", "problem solving", "technical skills", "precision", "building things",
            "independence", "trades", "engineering", "physical activity", "detail oriented",
            "critical thinking", "responsibility", "customer service", "tools", "systems thinking",
            "making things", "craftsmanship", "reliability", "planning", "community"
        ]
    },
    {
        "field": "Automotive Mechanics & Technology",
        "emoji": "🔩",
        "description": "Diagnosing, repairing, and maintaining vehicles using mechanical and electronic expertise.",
        "tags": [
            "hands on", "problem solving", "technology", "engineering", "precision",
            "critical thinking", "detail oriented", "technical skills", "tools", "independence",
            "trades", "logic", "physical activity", "cars", "systems thinking",
            "craftsmanship", "reliability", "making things", "customer service", "innovation"
        ]
    },
    {
        "field": "Cleaning & Facilities Management",
        "emoji": "🧹",
        "description": "Maintaining safe, clean, and functional environments across homes, offices, and public spaces.",
        "tags": [
            "responsibility", "physical activity", "detail oriented", "reliability",
            "community", "service", "hands on", "organisation", "teamwork", "health",
            "independence", "routine", "care", "environment", "precision",
            "customer service", "planning", "safety", "hard work", "efficiency"
        ]
    },
    {
        "field": "Elderly Care & Gerontology",
        "emoji": "👴",
        "description": "Supporting the health, dignity, and quality of life of older adults in homes and care settings.",
        "tags": [
            "helping people", "empathy", "care", "patience", "compassion",
            "health", "community", "human behaviour", "listening", "service",
            "making a difference", "people skills", "medicine", "responsibility", "wellbeing",
            "teamwork", "resilience", "social impact", "hands on", "dignity"
        ]
    },

    # ── SCIENCE & ENGINEERING ────────────────────────────────────────────────

    {
        "field": "Mining & Resources Engineering",
        "emoji": "⛏️",
        "description": "Extracting and processing earth's natural resources safely, sustainably, and efficiently.",
        "tags": [
            "engineering", "science", "outdoors", "problem solving", "technology",
            "physical activity", "mathematics", "geology", "responsibility", "safety",
            "systems thinking", "precision", "teamwork", "economics", "innovation",
            "environment", "planning", "critical thinking", "logistics", "hands on"
        ]
    },
    {
        "field": "Chemical Engineering",
        "emoji": "⚗️",
        "description": "Transforming raw materials into useful products through chemistry, physics, and process design.",
        "tags": [
            "science", "chemistry", "engineering", "problem solving", "mathematics",
            "precision", "research", "critical thinking", "innovation", "systems thinking",
            "detail oriented", "data", "technology", "safety", "manufacturing",
            "analysis", "logic", "continuous learning", "future", "responsibility"
        ]
    },
    {
        "field": "Mechatronics & Robotics",
        "emoji": "🤖",
        "description": "Merging mechanical, electrical, and software engineering to build smart automated systems.",
        "tags": [
            "engineering", "technology", "problem solving", "innovation", "mathematics",
            "computers", "automation", "hands on", "critical thinking", "precision",
            "robotics", "coding", "systems thinking", "curiosity", "future",
            "logic", "building things", "detail oriented", "continuous learning", "creativity"
        ]
    },
    {
        "field": "Technical & Artisan Trades",
        "emoji": "🛠️",
        "description": "Applying specialist technical skills in electrical, welding, carpentry, and other skilled trades.",
        "tags": [
            "hands on", "technical skills", "craftsmanship", "problem solving", "precision",
            "building things", "tools", "physical activity", "independence", "detail oriented",
            "making things", "engineering", "reliability", "trades", "systems thinking",
            "creativity", "responsibility", "logic", "community", "hard work"
        ]
    },
    {
        "field": "Research & Development (R&D)",
        "emoji": "🔬",
        "description": "Pushing the boundaries of knowledge and turning scientific discoveries into real-world solutions.",
        "tags": [
            "research", "curiosity", "science", "innovation", "critical thinking",
            "problem solving", "data", "analysis", "continuous learning", "creativity",
            "mathematics", "technology", "precision", "experimentation", "future",
            "collaboration", "detail oriented", "discovery", "patience", "making a difference"
        ]
    },

    # ── MATHEMATICS & STATISTICS ─────────────────────────────────────────────

    {
        "field": "Mathematics & Pure Sciences",
        "emoji": "📐",
        "description": "Exploring abstract structures, proofs, and the fundamental rules that govern the universe.",
        "tags": [
            "mathematics", "logic", "critical thinking", "problem solving", "patterns",
            "precision", "research", "curiosity", "analysis", "numbers",
            "abstract thinking", "continuous learning", "detail oriented", "science", "theory",
            "independence", "creativity", "innovation", "academia", "discovery"
        ]
    },
    {
        "field": "Statistics & Actuarial Science",
        "emoji": "📉",
        "description": "Quantifying uncertainty, modelling risk, and turning numbers into strategic decisions.",
        "tags": [
            "mathematics", "statistics", "data", "patterns", "analysis",
            "precision", "problem solving", "logic", "research", "critical thinking",
            "risk", "economics", "numbers", "detail oriented", "strategy",
            "finance", "science", "decision making", "accuracy", "future"
        ]
    },
]


def _profile_text(profile):
    """Flatten a profile's tags into a single space-separated document."""
    return " ".join(profile["tags"])


def recommend(user_inputs, top_n=3):
    """
    Compare the user's selected interests against every career profile
    using TF-IDF vectorisation + Cosine Similarity, and return the
    top_n best-matching careers.

    Returns a list of dicts:
        field, emoji, description, score (0-1 float),
        match_percentage (display string), matched_inputs (list[str])
    """
    if not user_inputs:
        return []

    user_doc = " ".join(user_inputs)
    corpus = [_profile_text(p) for p in CAREER_PROFILES] + [user_doc]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    user_vector = tfidf_matrix[-1]
    profile_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(user_vector, profile_vectors).flatten()

    ranked_indices = similarities.argsort()[::-1][:top_n]

    results = []
    for idx in ranked_indices:
        profile = CAREER_PROFILES[idx]
        score = float(similarities[idx])

        # Which of the user's own inputs actually relate to this profile's tags
        matched = [
            ui for ui in user_inputs
            if any(
                ui.lower() in tag.lower() or tag.lower() in ui.lower()
                for tag in profile["tags"]
            )
        ]
        if not matched:
            matched = user_inputs[:3]

        results.append({
            "field": profile["field"],
            "emoji": profile["emoji"],
            "description": profile["description"],
            "score": score,
            "match_percentage": f"{round(score * 100)}%",
            "matched_inputs": matched
        })

    return results
