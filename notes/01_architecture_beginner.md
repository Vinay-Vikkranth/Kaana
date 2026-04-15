# How the Ka'ana Recommendation System Works
### A beginner's reference — written to be understood, not to impress

---

## Table of Contents

1. [The Big Picture — Two Phases](#1-the-big-picture--two-phases)
2. [Core Concept: What is an Embedding?](#2-core-concept-what-is-an-embedding)
3. [Core Concept: What is Normalization?](#3-core-concept-what-is-normalization)
4. [Core Concept: What is Cosine Similarity?](#4-core-concept-what-is-cosine-similarity)
5. [Core Concept: What is Qdrant?](#5-core-concept-what-is-qdrant)
6. [Phase 1 — Offline (Startup)](#6-phase-1--offline-startup)
7. [Phase 2 — Online (Per Request)](#7-phase-2--online-per-request)
8. [The Full Flow in One Picture](#8-the-full-flow-in-one-picture)
9. [Terminology Cheat Sheet](#9-terminology-cheat-sheet)
10. [My Own Notes](#10-my-own-notes)

---

## 1. The Big Picture — Two Phases

The entire system runs in two distinct phases:

```
PHASE 1 — OFFLINE
  Happens once, when the server starts up.
  Goal: convert 70 restaurant descriptions into 70 vectors stored in Qdrant.
  Files involved: seed_data.py → seed_runner.py → models.py → vector_store.py

PHASE 2 — ONLINE
  Happens every time a user clicks "Get My Matches".
  Goal: convert the user's Taste ID into a vector, find the closest restaurant vectors.
  Files involved: models.py → vector_store.py → recommender.py → main.py
```

Think of Phase 1 as doing all the heavy preparation work once, so Phase 2 can be
lightning fast (under 30ms) for every user.

### The Ka'ana "Taste ID" concept

Ka'ana is built around the idea that **5 dimensions cover 80% of what makes a
recommendation great** (the Pareto principle — 20% of inputs drive 80% of the results).

Those 5 dimensions are:

| # | Dimension | Example choices |
|---|-----------|-----------------|
| 1 | **Dietary** | no restrictions / vegetarian / vegan / pescatarian / gluten-free |
| 2 | **Occasion** | romantic date / friends group / family & kids / solo explorer / special occasion / adventurous |
| 3 | **Cuisine** | local Hawaiian / Japanese Pacific / modern fusion / comfort casual / healthy light / world cuisine |
| 4 | **Atmosphere** | beach outdoor / upscale elegant / lively social / cozy local / scenic unique |
| 5 | **Budget** | $ / $$ / $$$ / $$$$ |

Two additional practical signals:
- **Noise preference** (1 = quiet conversation, 4 = loud rowdy)
- **Current time** (so we can tell if a restaurant is open right now)

---

## 2. Core Concept: What is an Embedding?

### The simple idea

You can't do math on words. You CAN do math on numbers.
An **embedding** is how we convert meaning into numbers.

Imagine describing a restaurant to someone using only numbers:
```
How spicy?       → 7 out of 10
How expensive?   → 3 out of 10
How casual?      → 8 out of 10
How beachy?      → 9 out of 10
```
That list `[7, 3, 8, 9]` is an embedding. Four numbers encode the personality of a restaurant.

In our system, instead of 4 human-chosen numbers, we use **384 numbers** produced by
a neural network (SBERT). These 384 numbers are not human-readable — they are abstract
learned dimensions. But together they encode the full meaning of any text.

```
"romantic upscale beachfront fresh seafood quiet dinner Maui"
              │
              ▼
    SBERT (pre-trained neural network)
              │
              ▼
  [0.12, -0.34, 0.71, 0.08, -0.22, ...]
   ← 384 numbers that encode the meaning →
```

### Why 384 and not some other number?

384 is the output size of the specific model we use: `all-MiniLM-L6-v2`.
Different models produce different sizes (768, 1024, 1536...).
The number itself doesn't matter — what matters is that similar texts produce similar lists.

### The key property

> Texts with similar meanings produce similar lists of numbers.
> Texts with very different meanings produce very different lists of numbers.

So "fresh ahi poke bowl" and "fiery raw tuna Hawaiian street food" will produce nearly
identical 384-number lists, even though they don't share the same words.
This is the superpower that makes the whole system work.

---

## 3. Core Concept: What is Normalization?

### The problem it solves

Two lists of numbers can point in the same "direction" but have very different sizes:
```
Mama's Fish House: [6, 3, 8, 9]     total length = 50.2
Tin Roof Maui:     [3, 1.5, 4, 4.5] total length = 25.1
```
These are proportionally identical (Tin Roof is just Mama's divided by 2), but a computer
comparing them without normalization would think they're very different because of scale.

### What normalization does

It rescales every vector so its total length equals exactly **1.0**, regardless of how
big it was before:
```
Before:  [6, 3, 8, 9]       length = 50.2  ← different scales
Before:  [3, 1.5, 4, 4.5]   length = 25.1

After:   [0.56, 0.28, 0.75, 0.84]   length = 1.0  ← same scale ✓
After:   [0.56, 0.28, 0.75, 0.84]   length = 1.0  ← identical ✓
```

### The globe analogy

Imagine a globe. Every restaurant is a pin stuck on the surface. Normalization is what
forces every pin to sit exactly on the surface — not floating inside, not floating outside.

```
                ● Mama's Fish House  (romantic fine dining, Maui)
           ●                    ●
      ●  Nobu Honolulu        The Beach House    ●
    (upscale japanese)       (romantic oceanfront)

 ●                                                  ●
Giovanni's Shrimp            Helena's Hawaiian Food
(cheap casual shrimp truck)  (authentic local Oahu)

      ●                        ●
           ●              ●
                ● Tin Roof  (cheap local plate lunch)
```

Restaurants with similar meaning cluster together. Very different restaurants sit far
apart. This globe is called a **unit sphere** (a sphere with radius 1.0).

### The math (L2 formula)

```
length = √(n1² + n2² + n3² + ... + n384²)
normalized number = original number ÷ length
```

In our code (`models.py`), we use `normalize_embeddings=True` inside SBERT's
`.encode()` call, which does this automatically.

---

## 4. Core Concept: What is Cosine Similarity?

Once everything is on the unit sphere, how do you measure how close two pins are?

You measure the **angle between them** from the center of the globe.

```
User wants → "romantic beachfront upscale pescatarian Maui dinner" ──────●

                ● The Beach House     ← small angle = very similar
           ● Mama's Fish House        ← small angle = similar
                                ● Da Kitchen     ← big angle = different
                                        ● Giovanni's ← very big angle = very different
```

### The scoring

| Angle  | Cosine Similarity | Meaning              |
|--------|------------------|----------------------|
| 0°     | 1.0              | Perfect match        |
| 45°    | 0.7              | Strong similarity    |
| 90°    | 0.0              | Completely unrelated |
| 180°   | -1.0             | Total opposites      |

### Why "cosine"?

The mathematical function that measures angle is called cosine. The cosine of 0° is 1.
The cosine of 90° is 0. The cosine of 180° is -1.

### The shortcut (why normalization matters here)

Because all vectors are normalized to length 1.0, cosine similarity reduces to
just a **dot product** — multiply each pair of numbers and add them all up:

```python
cosine_similarity = v1[0]×v2[0] + v1[1]×v2[1] + ... + v1[383]×v2[383]
```

This is one of the cheapest math operations possible. Qdrant can do this for
all 70 restaurants in microseconds.

---

## 5. Core Concept: What is Qdrant?

### The difference from a regular database

A regular database (like Excel or MySQL) stores rows and searches by exact match:
```
"Find all restaurants where price = 2 AND island = 'Maui'"
→ exact matching only
```

Qdrant is a **vector database**. It stores lists of numbers (vectors) and searches by
*similarity* — finding the vectors whose numbers are closest to a query vector:
```
"Find the 50 restaurant vectors closest to THIS user vector"
→ similarity matching in mathematical space
```

You cannot do this with MySQL. It was built specifically for this problem.

### What Qdrant stores per restaurant

```
┌─────────────────────────────────────────────────────────────┐
│  Entry #17                                                  │
│                                                             │
│  VECTOR  →  [0.12, -0.34, 0.71, 0.08, ...]                 │
│             384 numbers — used for SEARCHING                │
│                                                             │
│  PAYLOAD →  { "name": "Mama's Fish House",                  │
│               "island": "maui",                             │
│               "price_range": 4,                             │
│               "dietary_tags": ["seafood_heavy"],            │
│               "occasion_tags": ["romantic", "special_occasion"], │
│               "open_hours": {"open": 11, "close": 21} }    │
│             Full restaurant data — used for DISPLAY         │
└─────────────────────────────────────────────────────────────┘
```

- The **vector** is how you FIND it (math).
- The **payload** is what you SHOW the user (content).

### What does `:memory:` mean?

```python
QdrantClient(":memory:")   # in vector_store.py
```

| Mode | Normal Qdrant | Our POC (`:memory:`) |
|------|--------------|---------------------|
| Data saved to | Hard drive | RAM only |
| Survives restart? | Yes | No (rebuilt in ~5s) |
| Requires Docker? | Yes | No |
| Setup needed? | Yes | Zero |

For a POC this is ideal. When moving to production, switch to a real Qdrant
server (or Pinecone, Weaviate, pgvector) — the rest of the code stays the same.

---

## 6. Phase 1 — Offline (Startup)

**File flow:** `seed_data.py` → `seed_runner.py` → `models.py` → `vector_store.py`

### The restaurant data

70 curated real Hawaii restaurants across all 4 islands:
```
Oahu:       22 restaurants (Honolulu, Waikiki, North Shore, Kaimuki...)
Maui:       22 restaurants (Kihei, Lahaina area, Paia, Wailea, Kapalua...)
Big Island: 14 restaurants (Kona, Hilo, Waimea, Kohala...)
Kauai:      12 restaurants (Poipu, Hanalei, Kapaa, Lihue...)
```

Note: Lahaina Front Street venues were excluded — they were destroyed in the
August 2023 wildfire.

Each restaurant has new fields beyond just tags:
- `island` — which island it's on
- `dietary_tags` — e.g. `["vegetarian_friendly", "gluten_free_options"]`
- `occasion_tags` — e.g. `["romantic", "special_occasion", "social"]`

### Step 1 — Restaurant dict becomes a sentence
*File: `models.py` — `RestaurantEncoder._build_text()`*

Numbers get translated into words so SBERT can understand them:
```
price_range = 2   →   "moderate affordable mid-range"
noise_level = 3   →   "lively buzzy energetic social"
spice_level = 2   →   "medium lightly seasoned"
island = "maui"   →   "Maui Paia Kihei Wailea Kapalua Hana upcountry island"
dietary_tags      →   "seafood_heavy vegetarian_friendly"
occasion_tags     →   "romantic special_occasion social"
tags              →   "oceanfront fresh fish beachfront seafood"
```

All fields are joined into one rich paragraph:
```
"Mama's Fish House. Fresh Seafood. Island: Maui Paia Kihei Wailea...
 Cuisine style: seafood. Vibe: romantic. Tags: oceanfront fresh fish...
 Dietary: seafood_heavy. Best for: romantic special_occasion.
 Price: upscale splurge pricey. Atmosphere: quiet relaxed easy conversation.
 Spice: mild gentle. World-famous romantic dinner spot in a thatched-roof
 plantation house steps from the ocean..."
```

> **Why translate numbers to words?**
> SBERT understands language, not raw integers. `noise_level=3` means nothing to it.
> `"lively buzzy energetic social"` carries full meaning.

### Step 2 — Sentence becomes 384 numbers
*File: `models.py` — `self.sbert.encode(..., normalize_embeddings=True)`*

SBERT reads the paragraph and produces 384 numbers. The `normalize_embeddings=True`
flag L2-normalizes the result so the vector sits on the unit sphere.

### Step 3 — Vector goes into Qdrant
*File: `vector_store.py` — `upsert_restaurant()`*

The 384-number vector and the full restaurant dict are stored together in Qdrant.
This repeats for all 70 restaurants. Startup is now complete.

---

## 7. Phase 2 — Online (Per Request)

**File flow:** `main.py` → `models.py` → `vector_store.py` → `recommender.py` → browser

### Step 4 — Taste ID answers become a sentence
*File: `models.py` — `UserEncoder._build_text()`*

Each Taste ID selection maps to a descriptive phrase:
```python
dietary = "pescatarian"  →  "pescatarian seafood fish no red meat no poultry vegetables"
occasion = "romantic_date" → "romantic intimate quiet dinner date special occasion candlelight couple"
cuisine = "japanese_pacific" → "Japanese sushi sashimi ramen poke Pacific Asian noodles umami fresh fish"
atmosphere = "upscale_elegant" → "elegant upscale fine dining white tablecloth quiet intimate slow paced formal"
budget = 3               →  "upscale splurge forty to eighty dollars nice restaurant"
noise_pref = 2           →  "quiet conversation easy talk relaxed moderate"
island = "maui"          →  "Maui Paia Kihei Wailea Kapalua Hana upcountry island"
```

These are joined into one rich sentence describing what the user wants.

### Step 5 — User sentence becomes a 384-D vector
*File: `models.py` — `UserEncoder.encode()`*

The **exact same SBERT model** is used. Same model = same 384-D space.
This is why user and restaurant vectors are directly comparable.

### Step 6 — Qdrant finds 50 closest restaurants
*File: `vector_store.py` — `query()`*

Qdrant computes dot product between the user vector and all 70 restaurant vectors
simultaneously. Returns the 50 with highest cosine similarity scores.

We pull 50 (not just 10) because some will be eliminated by hard filters next.

### Step 7 — Hard filters remove impossible matches
*File: `recommender.py` + `vector_store.py`*

Two rounds of non-negotiable rules applied after the vector search:

**Round 1 — Dietary (in `recommender.py`):**
```
user is vegan → remove any restaurant without "vegan_options" in dietary_tags
user is gluten_free → remove any restaurant without "gluten_free_options"
```

**Round 2 — Price + Hours (in `vector_store.py` → `apply_hard_filters()`):**
```
restaurant price_range > user's budget → removed
restaurant is currently closed at user's current_time → removed
```

A perfect semantic match still gets cut if it fails these checks.
Safety fallback: if ALL restaurants are filtered out, restore the previous set.

### Step 8 — Late fusion blends two scores
*File: `recommender.py` — `score_match()`*

```
cosine_score   = dot product of user vector and restaurant vector (0.0 – 1.0)

attr_score  (weights sum to 1.0):
  + 0.30  if restaurant price_range ≤ user budget
  + 0.20  if restaurant is open at user's current_time
  + 0.20  if occasion tags match (e.g. "romantic" ↔ "romantic_date")
  + 0.15  if noise_level within ±1 of noise_pref
  + 0.15  if restaurant is on the user's selected island
  ──────
  0.0 to 1.0

final_score = (cosine_score × 0.65) + (attr_score × 0.35)
```

The semantic score gets more weight (65%) because it captures nuanced meaning.
The attribute score (35%) corrects for practical constraints.

### Step 9 — Explainability generates plain-English reasons
*File: `recommender.py` — `generate_explanation()`*

Rule-based system that looks at user profile + restaurant attributes and produces
human-readable bullet points:
```python
if dietary == "pescatarian" AND "seafood_heavy" in restaurant dietary_tags:
    → "Seafood-forward menu fits your pescatarian preference"

if occasion == "romantic_date" AND "romantic" in restaurant occasion_tags:
    → "Ideal setting for a romantic dinner"

if atmosphere == "beach_outdoor" AND "beachfront" in restaurant tags:
    → "The beach and ocean setting you're looking for"

if cosine_score > 0.80:
    → "Exceptional taste match (83% similarity)"
```

### Step 10 — Top 10 returned to browser

Final sorted list sent as JSON. Browser renders cards, score bars (blue = semantic 65%,
orange = attribute 35%), why-reasons, description, top-3 comparison table.

---

## 8. The Full Flow in One Picture

```
SERVER STARTUP (once, ~5 seconds)
──────────────────────────────────────────────────────────────────────
seed_data.py         models.py                    Qdrant (RAM)
┌────────────┐       ┌──────────────────────────┐  ┌────────────────┐
│ 70 dicts   │──────▶│ RestaurantEncoder         │  │                │
│            │       │                          │  │ 70 vectors     │
│ {name,     │       │ 1. dict → rich sentence  │─▶│ + payloads     │
│  island,   │       │ 2. SBERT → 384-D vector  │  │                │
│  tags,     │       │ 3. normalize (len = 1.0) │  │ oahu: 22       │
│  dietary_  │       └──────────────────────────┘  │ maui: 22       │
│  tags,     │                                     │ big_isl: 14    │
│  occasion_ │                                     │ kauai: 12      │
│  tags...}  │                                     └────────────────┘
└────────────┘

PER REQUEST (~30ms)
──────────────────────────────────────────────────────────────────────
Browser              models.py             vector_store.py  recommender.py
┌────────────────┐   ┌─────────────────┐   ┌─────────────┐  ┌───────────────┐
│ Taste ID       │   │ UserEncoder     │   │             │  │               │
│                │   │                 │   │ dot product │  │ 0.65×cosine   │
│ dietary        │──▶│ 5 answers       │──▶│ vs all 70   │─▶│ + 0.35×attr   │
│ occasion       │   │ → sentence      │   │             │  │               │
│ cuisine        │   │ → SBERT 384-D   │   │ top 50      │  │ hard filter   │
│ atmosphere     │   │ → normalize     │   │             │  │ dietary       │
│ budget         │   └─────────────────┘   │ price +     │  │ price         │
│ noise / time   │                         │ hours       │  │ hours         │
│ island         │                         │ filter      │  │               │
└────────────────┘                         └─────────────┘  │ explanations  │
       ▲                                                     │               │
       └──────────────── top 10 JSON ◀───────────────────── └───────────────┘
```

---

## 9. Terminology Cheat Sheet

| Term | Plain English |
|------|--------------|
| **Embedding** | A list of numbers that represents the *meaning* of something |
| **Vector** | Same as embedding — just a list of numbers |
| **384-D** | The list has 384 numbers. "D" stands for dimensions |
| **SBERT** | The AI model (`all-MiniLM-L6-v2`) that converts text into 384 numbers |
| **Normalize / L2 normalization** | Rescale every vector so its total length = exactly 1.0 |
| **Unit sphere** | The imaginary globe where all normalized vectors live |
| **Cosine similarity** | The angle between two vectors — measures similarity (0.0 to 1.0) |
| **Dot product** | Multiply each pair of numbers and sum them — the math behind cosine similarity |
| **Qdrant** | A vector database that stores vectors and searches by similarity |
| **`:memory:`** | Keep database in RAM only — no disk, no Docker, lost on restart |
| **Payload** | The full restaurant dict stored alongside the vector in Qdrant |
| **Two-tower** | Two separate encoders (restaurant + user) that produce vectors in the same space |
| **Late fusion** | Combining semantic score and attribute score at the end of the pipeline |
| **Hard filter** | A non-negotiable rule — if dietary restriction not met, remove regardless of score |
| **Upsert** | "Update or insert" — store a vector, overwrite if ID already exists |
| **Seed / Seeding** | Encoding all 70 restaurants and loading them into Qdrant at startup |
| **Taste ID** | Ka'ana's 5-dimension user profile: dietary + occasion + cuisine + atmosphere + budget |
| **Pareto principle** | The idea that 5 well-chosen dimensions (20% of inputs) drive 80%+ of accuracy |
| **dietary_tags** | Tags on each restaurant indicating dietary compatibility (e.g. `vegan_options`) |
| **occasion_tags** | Tags on each restaurant indicating best use cases (e.g. `romantic`, `family`) |

---

## 10. My Own Notes

> *Add your own notes, questions, and observations here as you learn more.*

---
