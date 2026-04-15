# models.py — Two-tower encoders using raw SBERT 384-D output
#
# WHY NO PROJECTION LAYER:
# A 384→128 linear projection only improves quality when TRAINED on user
# interaction data (clicks, orders, skips). Without that signal it is a
# random matrix that degrades already-good SBERT vectors. Both towers share
# the same SentenceTransformer so they already live in the same 384-D space.

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 native output size

# ── Taste ID dimension mappings ────────────────────────────────────────────
# These map every onboarding answer to a rich semantic phrase. The richer
# the phrase, the better SBERT can place it in the correct neighbourhood
# of the embedding space relative to restaurant descriptions.

DIETARY_MAP = {
    "no_restrictions":    "any cuisine all proteins meat fish vegetarian everything",
    "vegetarian":         "vegetarian no meat plant based vegetables tofu eggs dairy",
    "vegan":              "vegan no animal products plant based vegetables fruits nuts grains",
    "pescatarian":        "pescatarian seafood fish no red meat no poultry vegetables",
    "gluten_free":        "gluten free no wheat no gluten rice based naturally gluten free",
}

OCCASION_MAP = {
    "romantic_date":      "romantic intimate quiet dinner date special occasion candlelight couple",
    "friends_group":      "group of friends social loud fun lively drinks sharing plates",
    "family_kids":        "family friendly kids children casual comfortable familiar food",
    "solo_explorer":      "solo dining quick comfortable counter service simple focused eating",
    "special_occasion":   "celebration anniversary birthday milestone upscale memorable evening",
    "adventurous":        "adventurous new experience unique unexpected discovery bold food",
}

CUISINE_MAP = {
    "local_hawaiian":     "traditional Hawaiian local plate lunch poke kalua pig poi luau authentic cultural",
    "japanese_pacific":   "Japanese sushi sashimi ramen poke Pacific Asian noodles umami fresh fish",
    "modern_fusion":      "modern fusion creative innovative chef driven contemporary technique upscale",
    "comfort_casual":     "comfort food casual burgers pizza tacos familiar hearty satisfying simple",
    "healthy_light":      "healthy light fresh salad bowls organic clean eating vegetable forward",
    "world_cuisine":      "international world cuisine Thai Indian Mediterranean Italian global flavors",
}

ATMOSPHERE_MAP = {
    "beach_outdoor":      "beachfront outdoor oceanfront toes in sand tiki casual surf tropical open air",
    "upscale_elegant":    "elegant upscale fine dining white tablecloth quiet intimate slow paced formal",
    "lively_social":      "lively bar social buzzy energetic drinks cocktails music fun loud",
    "cozy_local":         "cozy local neighborhood rustic casual unpretentious community simple",
    "scenic_unique":      "scenic view unique experience special setting memorable location beautiful",
}

BUDGET_MAP = {
    1: "budget cheap affordable under fifteen dollars counter service takeout",
    2: "moderate mid-range casual dining twenty to forty dollars",
    3: "upscale splurge forty to eighty dollars nice restaurant",
    4: "fine dining luxury over eighty dollars tasting menu special occasion",
}

NOISE_MAP = {
    1: "very quiet peaceful silent library hushed whisper",
    2: "quiet conversation easy talk relaxed moderate",
    3: "lively energetic buzzy music background noise social",
    4: "loud rowdy bar music live entertainment noisy vibrant",
}

ISLAND_MAP = {
    "oahu":       "Oahu Honolulu Waikiki North Shore Kaimuki Hawaii island",
    "maui":       "Maui Paia Kihei Wailea Kapalua Hana upcountry island",
    "big_island": "Big Island Kona Hilo Waimea Kohala Hawaii island volcano",
    "kauai":      "Kauai Poipu Hanalei Kapaa Lihue garden island Hawaii",
    "any":        "Hawaii island any location all islands",
}


class RestaurantEncoder:
    """Encodes a restaurant dict into a 384-D L2-normalized unit vector."""

    def __init__(self):
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")

    def _build_text(self, r: dict) -> str:
        price_labels  = {1: "budget cheap affordable", 2: "moderate affordable mid-range",
                         3: "upscale splurge pricey", 4: "fine dining expensive luxury tasting menu"}
        noise_labels  = {1: "very quiet peaceful intimate", 2: "quiet relaxed easy conversation",
                         3: "lively buzzy energetic social", 4: "loud rowdy bar music vibrant"}
        spice_labels  = {1: "mild gentle not spicy", 2: "medium lightly seasoned",
                         3: "spicy bold heat chili", 4: "very spicy fiery intense heat"}

        dietary_text  = " ".join(r.get("dietary_tags",  []))
        occasion_text = " ".join(r.get("occasion_tags", []))
        tags_text     = " ".join(r.get("tags", []))
        island_text   = ISLAND_MAP.get(r.get("island", ""), "")

        parts = [
            r.get("name", ""),
            r.get("cuisine", ""),
            f"Island: {island_text}",
            f"Cuisine style: {r.get('cuisine_style', '')}",
            f"Vibe: {r.get('vibe', '')}",
            f"Tags: {tags_text}",
            f"Dietary: {dietary_text}",
            f"Best for: {occasion_text}",
            f"Price: {price_labels.get(r.get('price_range', 2), '')}",
            f"Atmosphere: {noise_labels.get(r.get('noise_level', 2), '')}",
            f"Spice: {spice_labels.get(r.get('spice_level', 1), '')}",
            r.get("description", ""),
        ]
        return ". ".join(p for p in parts if p)

    def encode(self, restaurant: dict) -> np.ndarray:
        text = self._build_text(restaurant)
        vec  = self.sbert.encode(text, convert_to_tensor=False, normalize_embeddings=True)
        return vec.astype(np.float32)


class UserEncoder:
    """
    Encodes a Taste ID profile into a 384-D L2-normalized unit vector.
    Uses the same SentenceTransformer as RestaurantEncoder — same 384-D space.

    Taste ID keys (aligned with Todd's 80/20 Pareto dimensions):
      dietary      : no_restrictions | vegetarian | vegan | pescatarian | gluten_free
      occasion     : romantic_date | friends_group | family_kids | solo_explorer |
                     special_occasion | adventurous
      cuisine      : local_hawaiian | japanese_pacific | modern_fusion |
                     comfort_casual | healthy_light | world_cuisine
      atmosphere   : beach_outdoor | upscale_elegant | lively_social |
                     cozy_local | scenic_unique
      budget       : int 1–4
      noise_pref   : int 1–4
      island       : oahu | maui | big_island | kauai | any
      current_time : int 0–23
    """

    def __init__(self):
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2")

    def _build_text(self, p: dict) -> str:
        parts = [
            "I am looking for a restaurant.",
            DIETARY_MAP.get(p.get("dietary", "no_restrictions"), ""),
            OCCASION_MAP.get(p.get("occasion", "friends_group"), ""),
            CUISINE_MAP.get(p.get("cuisine", "local_hawaiian"), ""),
            ATMOSPHERE_MAP.get(p.get("atmosphere", "beach_outdoor"), ""),
            f"Budget: {BUDGET_MAP.get(p.get('budget', 2), '')}",
            f"Noise preference: {NOISE_MAP.get(p.get('noise_pref', 2), '')}",
            f"Island: {ISLAND_MAP.get(p.get('island', 'any'), '')}",
        ]
        return " ".join(part for part in parts if part)

    def encode(self, user_profile: dict) -> np.ndarray:
        text = self._build_text(user_profile)
        vec  = self.sbert.encode(text, convert_to_tensor=False, normalize_embeddings=True)
        return vec.astype(np.float32)
