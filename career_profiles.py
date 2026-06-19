"""
Career Profiles Dataset
------------------------
Each career field is represented as a profile with descriptive tags.
Tags cover: interests, values, work styles, personality traits, and themes.
This shared vocabulary is what makes cosine similarity work accurately.
"""

CAREER_PROFILES = [
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
            "computers", "continuous learning", "automation", "future", "science", "precision"
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
        "description": "Creating ventures, managing organizations, and driving economic value.",
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
        "description": "Managing money, analyzing financial systems, and enabling economic decisions.",
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
    }
]
