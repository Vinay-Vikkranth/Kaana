# recommender.py — Scoring, explainability, and recommend()

import numpy as np
from vector_store import is_open, apply_hard_filters


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _island_matches(restaurant: dict, user_profile: dict) -> bool:
    pref = user_profile.get("island", "any")
    if pref == "any":
        return True
    return restaurant.get("island", "") == pref


def _dietary_compatible(restaurant: dict, user_profile: dict) -> bool:
    """Hard check — if user has a dietary restriction, restaurant must support it."""
    dietary = user_profile.get("dietary", "no_restrictions")
    if dietary == "no_restrictions":
        return True
    dtags = restaurant.get("dietary_tags", [])
    mapping = {
        "vegetarian":  "vegetarian_friendly",
        "vegan":       "vegan_options",
        "pescatarian": "seafood_heavy",
        "gluten_free": "gluten_free_options",
    }
    required_tag = mapping.get(dietary, "")
    return required_tag in dtags


def _occasion_matches(restaurant: dict, user_profile: dict) -> bool:
    occasion = user_profile.get("occasion", "")
    otags = restaurant.get("occasion_tags", [])
    mapping = {
        "romantic_date":   ["romantic"],
        "friends_group":   ["social", "friends_group"],
        "family_kids":     ["family"],
        "solo_explorer":   ["solo"],
        "special_occasion":["romantic", "special_occasion"],
        "adventurous":     ["adventurous_dining", "social"],
    }
    compatible = mapping.get(occasion, [])
    return any(t in otags for t in compatible)


def score_match(user_vector, user_profile, restaurant, restaurant_vector):
    """
    Returns (final_score, cosine_score, attr_score).

    Weights (aligned with Todd's Pareto principle):
      65% semantic cosine similarity  — captures nuanced taste meaning
      35% structured attribute match  — price, occasion, island, noise, hours
    """
    cosine_score = max(0.0, cosine_similarity(user_vector, restaurant_vector))

    attr = 0.0
    # Price fit (most predictive hard constraint) — 0.30
    attr += 0.30 if restaurant.get("price_range", 4) <= user_profile.get("budget", 4) else 0.0
    # Open now — 0.20
    attr += 0.20 if is_open(restaurant, user_profile.get("current_time", 12)) else 0.0
    # Occasion match — 0.20
    attr += 0.20 if _occasion_matches(restaurant, user_profile) else 0.0
    # Noise / atmosphere proximity — 0.15
    noise_diff = abs(restaurant.get("noise_level", 2) - user_profile.get("noise_pref", 2))
    attr += 0.15 if noise_diff <= 1 else 0.0
    # Island match — 0.15
    attr += 0.15 if _island_matches(restaurant, user_profile) else 0.0

    final = round((cosine_score * 0.65) + (attr * 0.35), 4)
    return final, round(cosine_score, 4), round(attr, 4)


def generate_explanation(user_profile, restaurant, cosine_score, attr_score):
    reasons   = []
    tags      = [t.lower() for t in restaurant.get("tags", [])]
    dtags     = restaurant.get("dietary_tags", [])
    otags     = restaurant.get("occasion_tags", [])
    island    = restaurant.get("island", "")

    island_names = {"oahu": "Oahu", "maui": "Maui",
                    "big_island": "Big Island", "kauai": "Kauai"}

    # ── Dietary ──────────────────────────────────────────────────────────────
    dietary = user_profile.get("dietary", "no_restrictions")
    if dietary == "vegetarian" and "vegetarian_friendly" in dtags:
        reasons.append("Vegetarian-friendly menu")
    if dietary == "vegan" and "vegan_options" in dtags:
        reasons.append("Has vegan options")
    if dietary == "gluten_free" and "gluten_free_options" in dtags:
        reasons.append("Gluten-free options available")
    if dietary == "pescatarian" and "seafood_heavy" in dtags:
        reasons.append("Seafood-forward menu fits your pescatarian preference")

    # ── Occasion ─────────────────────────────────────────────────────────────
    occasion = user_profile.get("occasion", "")
    if occasion == "romantic_date" and "romantic" in otags:
        reasons.append("Ideal setting for a romantic dinner")
    if occasion == "friends_group" and any(t in otags for t in ["social", "friends_group"]):
        reasons.append("Great energy for a group night out")
    if occasion == "family_kids" and "family" in otags:
        reasons.append("Family-friendly and welcoming to kids")
    if occasion == "special_occasion" and any(t in otags for t in ["special_occasion", "romantic"]):
        reasons.append("Worthy of a special occasion")
    if occasion == "adventurous" and "adventurous_dining" in otags:
        reasons.append("Offers the adventurous dining experience you're after")
    if occasion == "solo_explorer" and "solo" in otags:
        reasons.append("Comfortable and welcoming for solo diners")

    # ── Cuisine culture ───────────────────────────────────────────────────────
    cuisine_pref = user_profile.get("cuisine", "")
    if cuisine_pref == "local_hawaiian" and restaurant.get("cuisine_style") == "local":
        reasons.append("Authentic local Hawaiian cuisine")
    if cuisine_pref == "modern_fusion" and restaurant.get("cuisine_style") == "fusion":
        reasons.append("Creative modern fusion approach you enjoy")
    if cuisine_pref == "healthy_light" and any(k in tags for k in ["healthy", "organic", "fresh", "light"]):
        reasons.append("Light, fresh, and health-conscious menu")
    if cuisine_pref == "japanese_pacific" and any(k in tags for k in ["sushi", "japanese", "poke", "ramen"]):
        reasons.append("Japanese and Pacific flavors you're looking for")
    if cuisine_pref == "comfort_casual" and any(k in tags for k in ["comfort food", "burgers", "plate lunch", "casual"]):
        reasons.append("Exactly the comfort food energy you want")

    # ── Atmosphere ───────────────────────────────────────────────────────────
    atm = user_profile.get("atmosphere", "")
    if atm == "beach_outdoor" and any(k in tags for k in ["beach", "beachfront", "oceanfront", "outdoor"]):
        reasons.append("The beach and ocean setting you're looking for")
    if atm == "upscale_elegant" and any(k in tags for k in ["fine dining", "upscale", "elegant", "intimate"]):
        reasons.append("Upscale and elegant atmosphere")
    if atm == "lively_social" and any(k in tags for k in ["social", "bar", "live music", "cocktails", "lively"]):
        reasons.append("Lively, social atmosphere with great drinks")
    if atm == "cozy_local" and any(k in tags for k in ["local", "local favorite", "neighborhood", "cozy", "rustic"]):
        reasons.append("The cozy, local neighborhood feel you prefer")
    if atm == "scenic_unique" and any(k in tags for k in ["scenic", "views", "ocean views", "unique", "historic"]):
        reasons.append("A unique or scenic setting worth experiencing")

    # ── Island ───────────────────────────────────────────────────────────────
    if user_profile.get("island", "any") != "any" and _island_matches(restaurant, user_profile):
        reasons.append(f"Located on {island_names.get(island, island.title())}")

    # ── Semantic score ────────────────────────────────────────────────────────
    if cosine_score > 0.80:
        reasons.append(f"Exceptional taste match ({cosine_score:.0%} similarity)")
    elif cosine_score > 0.65:
        reasons.append(f"Strong taste match ({cosine_score:.0%} similarity)")
    elif cosine_score > 0.50:
        reasons.append(f"Good taste alignment ({cosine_score:.0%} similarity)")

    # ── Practical ─────────────────────────────────────────────────────────────
    if is_open(restaurant, user_profile.get("current_time", 12)):
        reasons.append("Open right now")

    if not reasons:
        reasons.append("Good overall match for your taste profile")

    noise_diff = abs(restaurant.get("noise_level", 2) - user_profile.get("noise_pref", 2))
    match_breakdown = {
        "semantic_similarity": round(cosine_score, 4),
        "price_match":    1.0 if restaurant.get("price_range", 4) <= user_profile.get("budget", 4) else 0.0,
        "occasion_match": 1.0 if _occasion_matches(restaurant, user_profile) else 0.0,
        "noise_match":    round(max(0.0, 1.0 - noise_diff / 3.0), 2),
        "open_now":       1.0 if is_open(restaurant, user_profile.get("current_time", 12)) else 0.0,
        "island_match":   1.0 if _island_matches(restaurant, user_profile) else 0.5,
        "overall_score":  round((cosine_score * 0.65) + (attr_score * 0.35), 4),
    }
    return reasons, match_breakdown


def recommend(user_profile: dict, vector_store, user_encoder) -> list[dict]:
    """Full pipeline: encode → retrieve → dietary filter → score → explain → top 10."""
    try:
        user_vector = user_encoder.encode(user_profile)
    except Exception as e:
        raise RuntimeError(f"UserEncoder failed: {e}") from e

    raw_results = vector_store.query(user_vector, top_k=50)

    # Hard filter 1: dietary restrictions (non-negotiable)
    dietary_ok = [r for r in raw_results if _dietary_compatible(r.payload, user_profile)]
    if not dietary_ok:
        dietary_ok = raw_results  # safety fallback

    # Hard filter 2: price and open hours
    filtered = apply_hard_filters(dietary_ok, user_profile)
    if not filtered:
        filtered = dietary_ok

    scored = []
    for result in filtered:
        restaurant        = result.payload
        restaurant_vector = np.array(result.vector)
        try:
            final, cosine, attr = score_match(user_vector, user_profile, restaurant, restaurant_vector)
            reasons, breakdown  = generate_explanation(user_profile, restaurant, cosine, attr)
        except Exception:
            continue
        scored.append({
            "restaurant":    restaurant,
            "final_score":   final,
            "cosine_score":  cosine,
            "attr_score":    attr,
            "explanation":   reasons,
            "match_breakdown": breakdown,
        })

    scored.sort(key=lambda x: x["final_score"], reverse=True)
    return scored[:10]
