# The Ka'ana Recommendation System — Fully Explained
### Written for someone with zero background in AI or engineering

---

## Before we start — what this document is

This document explains every single thing that happens inside the Ka'ana restaurant
recommendation system. It is written for someone who has never studied computer science,
machine learning, or AI. Every term is explained the first time it is used. Nothing is
assumed. If you have read this and still feel confused about something, that is a gap
in this document — not a gap in you.

---

## Part 1 — The Problem We Are Solving

### Why is restaurant recommendation hard?

Imagine you are a travel guide in Hawaii. A tourist walks up to you and says:

> "I want somewhere romantic, not too loud, I eat fish but no meat, somewhere on Maui,
> and I don't want to spend more than $60 per person."

You, as a human, can immediately understand what they want. You have been to Maui. You
know what "romantic" feels like. You know what "not too loud" means. You can picture the
kind of restaurant they are describing. You can recommend three options instantly.

Now imagine you had to program a computer to do this.

The computer does not "know" what romantic means. It does not "know" what loud means.
It has no experience of being at a restaurant. It only knows numbers and logic.

So the question is: **how do you teach a computer to understand what a person means,
not just what they typed?**

This is the core problem Ka'ana solves. And the solution is surprisingly elegant.

---

### Why existing approaches fall short

Most apps solve this with simple filters:

- Show restaurants that match island = "Maui"
- Show restaurants that match category = "Seafood"
- Show restaurants that have rating > 4.0

This is called **keyword matching** or **filter-based search**. It works fine for basic
cases. But it completely fails at meaning.

For example:
- A user says they want "light, fresh food" but the database stores the category as
  "California cuisine". Those two things mean the same thing but share zero words.
- A user says "date night vibes" but the database doesn't have a "date night" field —
  just tags like "romantic", "candles", "quiet".
- A user says they want "adventure eating" — how would a filter handle that?

Ka'ana's approach doesn't match words. It matches **meaning**. That is the key difference.

---

## Part 2 — The Big Idea: Turning Meaning into Math

### Numbers can represent anything

Here is the foundation of everything in this system. Read this carefully:

**A computer cannot understand language. But a computer is excellent at math.**

So the trick is: convert language into numbers. If you can turn the meaning of words
into numbers, a computer can suddenly do very powerful things with them — like measure
how similar two pieces of text are.

Let's start with a simple version of this idea.

Suppose you wanted to describe a restaurant using only 4 numbers:

```
How expensive?     → score from 1 to 10
How casual?        → score from 1 to 10
How beachy?        → score from 1 to 10
How spicy?         → score from 1 to 10
```

Now consider two restaurants:

```
Mama's Fish House:  [8, 3, 9, 2]   (expensive, formal, very beachy, mild)
Paia Fish Market:   [3, 9, 8, 3]   (cheap, very casual, beachy, mild)
```

Even with just 4 numbers, you can see they are somewhat similar (both beachy, both mild)
but differ on price and formality.

Now imagine you had **384 numbers** instead of 4. And instead of a human choosing those
numbers, a highly trained AI chose them based on reading millions of books, articles,
and restaurant reviews. Those 384 numbers would capture meaning in extraordinary detail.

That is exactly what happens in Ka'ana. Every restaurant gets described by 384 numbers.
Every user preference gets described by 384 numbers. And then the computer compares them.

---

### The 384-number list is called a "vector" or "embedding"

In this system you will see the words **vector** and **embedding** used constantly. They
mean the same thing: a list of numbers that represents the meaning of something.

```
"Romantic upscale beachfront seafood restaurant on Maui"
                        │
              passes through the AI model
                        │
              ▼
[0.12, -0.34, 0.71, 0.08, -0.22, 0.55, -0.19, 0.44, ...]
 ←                   384 numbers                      →
```

This list of 384 numbers is the "vector" or "embedding" for that restaurant description.

The numbers themselves are not human-readable. You cannot look at number #47 and say
"ah yes, that's the beachiness score". The numbers work as a team — together they encode
the full meaning of the text, in a way that only the AI model understands.

The critical property is this:

> **Two texts with similar meanings will produce similar lists of numbers.**
> **Two texts with very different meanings will produce very different lists of numbers.**

So "romantic beachfront dinner Maui" and "intimate oceanfront evening Maui" will produce
nearly identical lists of 384 numbers — even though they don't share many words.

And "romantic beachfront dinner Maui" and "loud burger joint in a strip mall" will
produce very different lists of numbers.

This is the superpower that makes the whole system work.

---

### What is SBERT? The AI model that does the conversion

The system that converts text into 384 numbers is called **SBERT**. Specifically, we use
a model called `all-MiniLM-L6-v2`.

Think of SBERT as a very sophisticated translator. It was trained on hundreds of millions
of sentences from the internet, books, Wikipedia, and more. During that training, it
learned which words and phrases carry similar meanings. It learned that "romantic" and
"intimate" are related. That "beachfront" and "oceanfront" are related. That "loud" and
"rowdy" are related.

After training, this knowledge is "frozen" inside the model — stored as billions of
mathematical weights. When you give it a piece of text, it uses all that learned
knowledge to produce the 384 numbers that best represent the meaning of that text.

We do not train SBERT ourselves. It comes pre-trained, like a dictionary that was
already written by someone else. We just use it.

In the code (`models.py`):
```python
self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
# ...
vec = self.sbert.encode("some text", normalize_embeddings=True)
# vec is now a list of 384 numbers
```

---

## Part 3 — Normalization: Putting Everything on the Same Scale

### The problem with different-sized lists

After SBERT converts text to numbers, different texts can produce lists with very
different "sizes". Think of size as the total magnitude of all the numbers combined.

Imagine two restaurants with these (simplified) vectors:
```
Restaurant A:  [6, 3, 8, 9]    → total size = large
Restaurant B:  [3, 1.5, 4, 4.5] → total size = half of A
```

Notice that Restaurant B is just Restaurant A divided by 2. They are pointing in the
exact same direction — they represent the same concept — but one is "louder" (bigger).
Without adjusting for this, a computer comparing them would think they are different.

### The solution: normalize everything to the same size

**Normalization** is the process of rescaling every vector so that its total size
becomes exactly 1.0. It does not change the direction the numbers are pointing —
only the scale.

```
Before normalization:
  Restaurant A: [6, 3, 8, 9]      → size ≈ 12.7
  Restaurant B: [3, 1.5, 4, 4.5]  → size ≈ 6.3

After normalization:
  Restaurant A: [0.47, 0.24, 0.63, 0.71]  → size = 1.0
  Restaurant B: [0.47, 0.24, 0.63, 0.71]  → size = 1.0
```

After normalization, they are identical — as they should be, since they represent the
same concept.

### The globe analogy

This is the best way to picture what normalization does.

Imagine a globe (like a planet Earth). Every restaurant is a pin stuck on the surface.
Normalization is the rule that says every pin **must sit on the surface** — not floating
above it, not buried inside it. Every pin must be at exactly radius = 1.0 from the center.

This globe is called the **unit sphere** in mathematics. "Unit" just means size = 1.

```
                 ● Mama's Fish House  ← romantic fine dining, Maui
            ●                   ●
       ●  Nobu Honolulu     The Beach House
    (upscale japanese)    (romantic oceanfront)
  ●                                          ●
 Helena's                               Tin Roof
 (authentic local)                    (cheap plate lunch)
       ●                        ●
            ●              ●
                 ● Giovanni's ← casual shrimp truck
```

Restaurants with similar meaning (and thus similar numbers) cluster close together on
the globe's surface. Very different restaurants sit far apart.

In code, we do normalization automatically:
```python
vec = self.sbert.encode(text, normalize_embeddings=True)
# normalize_embeddings=True → SBERT handles normalization for us
```

---

## Part 4 — Cosine Similarity: Measuring How Close Two Restaurants Are

Now that every restaurant and every user profile is a pin on the globe's surface, how
do we measure how similar two of them are?

We measure the **angle** between them, as seen from the center of the globe.

```
Center of globe ────────────────────────────────► User's preferences (pin)

       ● Mama's Fish House   ← very small angle = very similar to user
  ● The Beach House          ← small angle = similar
               ● Da Kitchen  ← large angle = different from user
                     ● Giovanni's ← very large angle = very different
```

A small angle = very similar.
A large angle = very different.

The mathematical function that measures this angle is called **cosine**. When we say
"cosine similarity", we mean: "what is the cosine of the angle between these two vectors?"

The scores come out like this:

| Angle | Cosine Score | What it means |
|-------|-------------|---------------|
| 0°    | 1.00        | Perfect match — identical direction |
| 45°   | 0.71        | Strong similarity |
| 60°   | 0.50        | Moderate similarity |
| 90°   | 0.00        | Completely unrelated |
| 180°  | -1.00       | Total opposites |

In practice, most of our scores land between 0.50 and 0.85 — which is a good range.
Scores don't reach 1.0 because no restaurant is a perfect literal clone of a user's
described preferences.

### Why normalization makes this easy

Remember how we set every vector's size to exactly 1.0? That was not just for tidiness.
It means that measuring the angle becomes trivially simple math:

```
cosine_similarity = v1[0] × v2[0]
                  + v1[1] × v2[1]
                  + v1[2] × v2[2]
                  + ... (all 384 pairs)
```

This is called a **dot product** — you multiply each pair of numbers and add the results.
On modern hardware, a computer can do this for thousands of restaurants in microseconds.
Without normalization, the formula would be far more expensive to compute.

---

## Part 5 — The Restaurant Data

Before the system can recommend anything, it needs to know about the restaurants.

We have **70 real Hawaii restaurants** spread across all four islands:
- Oahu: 22 restaurants (Honolulu, Waikiki, North Shore, Kaimuki...)
- Maui: 22 restaurants (Paia, Kihei, Wailea, Kapalua...)
- Big Island: 14 restaurants (Kona, Hilo, Waimea...)
- Kauai: 12 restaurants (Poipu, Hanalei, Kapaa...)

Note: Lahaina Front Street venues are intentionally excluded — that area was destroyed
in the August 2023 wildfire.

Each restaurant is stored as a dictionary (a collection of labeled facts):

```python
{
    "id":          "rest_001",
    "name":        "Mama's Fish House",
    "island":      "maui",
    "cuisine":     "Seafood / Pacific Rim",
    "cuisine_style": "seafood",
    "vibe":        "romantic",
    "price_range": 4,          # 1=cheap, 2=moderate, 3=upscale, 4=fine dining
    "noise_level": 1,          # 1=quiet, 2=relaxed, 3=lively, 4=loud
    "spice_level": 1,          # 1=mild, 2=medium, 3=spicy, 4=very spicy
    "open_hours":  {"open": 11, "close": 21},
    "tags":        ["oceanfront", "fresh fish", "romantic", "beachfront", "upscale"],
    "dietary_tags":["seafood_heavy", "gluten_free_options"],
    "occasion_tags":["romantic", "special_occasion"],
    "location":    {"lat": 20.9298, "lng": -156.3225},
    "description": "World-famous romantic dinner spot..."
}
```

Each field tells the system something different:
- `tags` — general atmosphere and food descriptors
- `dietary_tags` — dietary compatibility (used for hard filtering)
- `occasion_tags` — what the restaurant is best for (used for scoring)
- `open_hours` — used to check if it's currently open
- `price_range` — compared directly against the user's budget
- `noise_level` — compared against the user's noise preference

---

## Part 6 — The Taste ID: How We Capture What a User Wants

When a user opens Ka'ana, they do not type a sentence. They click buttons and move
sliders. Behind the scenes, their selections form what we call a **Taste ID** — a
structured profile of their dining preferences.

The Taste ID has 7 fields:

```python
{
    "dietary":      "pescatarian",    # dietary restriction or preference
    "occasion":     "romantic_date",  # what is this dinner for?
    "cuisine":      "japanese_pacific", # what kind of food?
    "atmosphere":   "upscale_elegant",  # what kind of setting?
    "budget":       3,                  # 1=$, 2=$$, 3=$$$, 4=$$$$
    "noise_pref":   2,                  # 1=quiet to 4=loud
    "island":       "maui",             # which island?
    "current_time": 19,                 # 7:00 PM → used to check open hours
}
```

These 7 fields are the "20% of inputs that drive 80% of recommendation quality" —
the Pareto principle applied to dining. Ka'ana focuses on getting these 7 right rather
than asking the user 30 questions.

---

## Part 7 — Converting Selections into Rich Text

Here is a key insight: the numbers a user picks (like `budget = 3`) mean nothing to
SBERT. SBERT understands language. So we need to translate every selection into a
descriptive phrase before passing it to SBERT.

This is done in `models.py` using translation tables:

```
dietary = "pescatarian"
  → "pescatarian seafood fish no red meat no poultry vegetables"

occasion = "romantic_date"
  → "romantic intimate quiet dinner date special occasion candlelight couple"

cuisine = "japanese_pacific"
  → "Japanese sushi sashimi ramen poke Pacific Asian noodles umami fresh fish"

atmosphere = "upscale_elegant"
  → "elegant upscale fine dining white tablecloth quiet intimate slow paced formal"

budget = 3
  → "upscale splurge forty to eighty dollars nice restaurant"

noise_pref = 2
  → "quiet conversation easy talk relaxed moderate"

island = "maui"
  → "Maui Paia Kihei Wailea Kapalua Hana upcountry island"
```

All of these phrases get joined into one long description of what the user wants:

```
"I am looking for a restaurant. pescatarian seafood fish no red meat no poultry
 vegetables. romantic intimate quiet dinner date special occasion candlelight couple.
 Japanese sushi sashimi ramen poke Pacific Asian noodles umami fresh fish. elegant
 upscale fine dining white tablecloth quiet intimate slow paced formal. Budget:
 upscale splurge forty to eighty dollars nice restaurant. Noise preference: quiet
 conversation easy talk relaxed moderate. Island: Maui Paia Kihei Wailea..."
```

This rich text is then passed to SBERT, which produces 384 numbers representing what
this person wants. The more descriptive and specific the phrases, the better SBERT
can encode the real meaning.

The same process happens for restaurants. The restaurant dictionary also gets converted
into a rich text paragraph before SBERT processes it:

```
"Mama's Fish House. Seafood / Pacific Rim. Island: Maui Paia Kihei Wailea Kapalua
 Hana upcountry island. Cuisine style: seafood. Vibe: romantic. Tags: oceanfront
 fresh fish romantic beachfront upscale. Dietary: seafood_heavy gluten_free_options.
 Best for: romantic special_occasion. Price: upscale splurge pricey. Atmosphere:
 quiet relaxed easy conversation. Spice: mild gentle not spicy. World-famous romantic
 dinner spot in a thatched-roof plantation house steps from the ocean..."
```

Both the restaurant and the user get the same type of rich text paragraph. Both pass
through the same SBERT model. Both get 384 numbers back. Both are normalized to size 1.0.

And because they went through the **same model**, their 384-number lists exist in the
same "universe" — they are directly comparable. This is the whole trick.

---

## Part 8 — The Two-Tower Architecture

The system has two "towers" — two encoders:

```
TOWER 1 (RestaurantEncoder)          TOWER 2 (UserEncoder)
─────────────────────────────        ─────────────────────────────
Takes: restaurant dictionary         Takes: Taste ID answers
Does:  convert to rich text          Does:  convert to rich text
       → SBERT → 384 numbers                → SBERT → 384 numbers
       → normalize to size 1.0              → normalize to size 1.0
Output: restaurant vector            Output: user vector

                    SAME SBERT MODEL
                          │
                          ▼
               Both vectors live in the
               same 384-D universe
               → they can be compared
```

Why is it called "two-tower"? Picture two literal towers side by side. Each tower
processes a different type of input (restaurants vs. users). But both towers are
connected at the top — they share the same SBERT model, so they produce vectors
in the same mathematical space.

Because both vectors live in the same space, measuring the cosine similarity between
them tells you how closely the user's preferences match a restaurant's personality.

### Why there is no "projection layer" (advanced note)

You might see in some AI systems a step where vectors get compressed:
`384 numbers → some math → 128 numbers`

This is called a projection layer. We deliberately do NOT use one. Here is why:

A projection layer only helps when it has been **trained on real user data** — like
what restaurants users actually clicked, ordered from, or skipped. With that training,
it can learn to compress in a way that keeps the important signals.

Without training data, a projection layer is just a random matrix of numbers. Multiplying
our carefully crafted 384-number vectors by a random matrix actively makes them worse.
It's like scrambling a detailed portrait and then trying to compare portraits.

We use the raw 384-number output directly. It is already excellent.

---

## Part 9 — Qdrant: The Vector Database

After we create 384-number vectors for all 70 restaurants, we need to store them
somewhere that can search them quickly.

A regular database (like Excel, MySQL, or Google Sheets) is designed for exact matches:
```
"Show me rows where island = 'Maui' AND price = 3"
```

This cannot search by similarity. It cannot answer "show me the restaurants whose
384-number vectors are closest to this user's 384-number vector".

That is what **Qdrant** does. It is a **vector database** — a special type of database
built specifically for storing and searching vectors by similarity.

Think of Qdrant like a very smart filing cabinet. Every file has two parts:

```
┌────────────────────────────────────────────────────────────────┐
│  File for "Mama's Fish House"                                  │
│                                                                │
│  VECTOR  →  [0.12, -0.34, 0.71, 0.08, -0.22, ...]            │
│             384 numbers — used for SEARCHING                   │
│                                                                │
│  PAYLOAD →  {"name": "Mama's Fish House",                      │
│              "island": "maui",                                 │
│              "price_range": 4,                                 │
│              "open_hours": {"open": 11, "close": 21},          │
│              "tags": ["oceanfront", "romantic", ...],          │
│              "description": "World-famous romantic..."}        │
│             Complete restaurant data — used for DISPLAY        │
└────────────────────────────────────────────────────────────────┘
```

- The **vector** is how you FIND the restaurant (the math part)
- The **payload** is what you SHOW the user after finding it (the content part)

When searching, you give Qdrant a user vector and ask: "Give me the 50 restaurants
whose vectors are most similar to this." Qdrant checks all 70 simultaneously and
returns the top 50 ranked by cosine similarity. This takes milliseconds.

### What `:memory:` means

In the code you will see:
```python
QdrantClient(":memory:")
```

The `:memory:` part means: keep everything in RAM (your computer's short-term memory)
instead of writing to a hard drive. When the server restarts, the database is gone —
but it gets rebuilt automatically in about 5 seconds. This is fine for a prototype.

In production (a real app with real users), you would switch to a proper Qdrant server
that saves to disk. Everything else in the code would stay the same.

---

## Part 10 — The Full Journey: What Happens When You Click the Button

Let's walk through every single step from the moment you click "Find my restaurants"
to the moment results appear on screen.

---

### Step 0 — Before you even open the app (happens once when server starts)

When the server first turns on, it runs `seed_runner.py`. This file:

1. Reads all 70 restaurant dictionaries from `seed_data.py`
2. For each restaurant:
   a. Builds a rich text paragraph describing it (`RestaurantEncoder._build_text()`)
   b. Passes that text through SBERT → gets 384 numbers (`sbert.encode()`)
   c. Normalizes the vector to size 1.0
   d. Stores the vector + the original restaurant data in Qdrant
3. Logs: "Kaana ready. 70 restaurants loaded across 4 islands."

This takes about 5 seconds. After this, the 70 vectors sit in Qdrant in RAM, ready to
be searched instantly. This never happens again until the server restarts.

---

### Step 1 — You click the button

Your browser (Chrome, Safari, etc.) sends your Taste ID selections to the server as
a small package of data (called a JSON object):

```json
{
  "dietary":      "pescatarian",
  "occasion":     "romantic_date",
  "cuisine":      "japanese_pacific",
  "atmosphere":   "upscale_elegant",
  "budget":       3,
  "noise_pref":   2,
  "island":       "maui",
  "current_time": 19
}
```

The browser sends this to the URL `/recommend` using a method called `POST`. This is
just how browsers and servers talk to each other over the internet.

---

### Step 2 — The server receives it (`main.py`)

The server (FastAPI) receives the data and validates it. It checks that `budget` is
between 1 and 4, `current_time` is between 0 and 23, etc. If something is invalid,
it returns an error. If everything looks good, it passes the data to the recommender.

This file is the "traffic controller" — it receives requests and sends them to the
right place, but it doesn't do the actual recommendation work.

---

### Step 3 — The user profile becomes 384 numbers (`models.py`)

The `UserEncoder` takes the Taste ID and:

1. Translates each selection into a rich phrase using the mapping tables:
   ```
   "pescatarian" → "pescatarian seafood fish no red meat..."
   "romantic_date" → "romantic intimate quiet dinner date..."
   ... (all 7 fields)
   ```

2. Joins all phrases into one long text:
   ```
   "I am looking for a restaurant. pescatarian seafood fish... romantic intimate
    quiet dinner... Japanese sushi sashimi... elegant upscale fine dining..."
   ```

3. Passes this text through SBERT → gets 384 numbers

4. Normalizes to size 1.0

Result: one vector of 384 numbers that mathematically encodes what this user wants.

---

### Step 4 — Find the 50 closest restaurants (`vector_store.py`)

The user vector goes into Qdrant. Qdrant compares it against all 70 restaurant vectors
using the dot product (the fast cosine similarity math from Part 4).

It returns the **top 50** restaurants ranked by similarity score.

Why 50 and not 10? Because the next step is filtering, and some restaurants will be
removed. If we only fetched 10 and then removed 5 due to dietary restrictions, we would
only have 5 results — not enough. Fetching 50 gives us plenty of buffer.

---

### Two important questions about Step 3 and Step 4

Before we go further, two natural questions come up at this point in the pipeline.
Both are worth answering in full.

---

#### Question A — What exactly are those 384 numbers? Can I read them? Can I validate them?

This is one of the most important questions you can ask about this system.

**The honest answer: no, you cannot read the 384 numbers and understand them.**

Here is why. Each of the 384 numbers does not correspond to any single human concept.
There is no "number 47 = beachiness score". There is no "number 201 = how romantic".
The 384 dimensions are abstract. They are the result of a neural network learning from
hundreds of millions of sentences, and they encode meaning in a way that is spread
across all 384 numbers simultaneously.

This is called a **distributed representation**. The meaning is not stored in any one
dimension — it is stored in the pattern formed by all of them together, like how a
song isn't stored in any single note but in the relationship between all the notes.

Think of it like this. Imagine you describe a restaurant to a friend and they process
what you said and form an impression. You cannot open their head and point to one neuron
and say "that neuron stores the beachiness". The impression is distributed across
billions of neurons working together. SBERT works the same way.

**Where do the 384 numbers come from originally?**

The number 384 is a design decision made by the team that built the `all-MiniLM-L6-v2`
model. They designed a neural network with a specific architecture — 6 layers, specific
layer sizes — and 384 is the size of the output of the last layer. It is a hyperparameter
(a setting chosen before training), not something that emerged from the data.

Why 384 and not 768 or 128? It is a balance. Larger vectors (768, 1024, 1536) capture
more nuance but are slower to compute and take more storage. Smaller vectors (128, 64)
are fast but lose information. 384 is in the sweet spot for a model this size.

**How do you validate the system then, if you can't read the numbers?**

This is the key question. Since the vectors are not directly interpretable, you validate
the system indirectly — through its behavior. Here are the four main methods:

---

**Method 1: Nearest-neighbor sanity checks**

Pick a restaurant you know well and manually check what comes back as most similar.

```
Example: encode "Mama's Fish House" (romantic, upscale, seafood, Maui)
         and find its 5 nearest neighbors in the vector space.

Good result:  [The Beach House, Merriman's Maui, Pacific'O, Ferraro's]
              → all romantic, seafood-forward, Maui upscale venues ✓

Bad result:   [Da Kitchen, Rainbow Drive-In, Giovanni's Shrimp]
              → casual, cheap, very different style ✗
```

If similar restaurants cluster together, the vectors are working. If random restaurants
end up near each other, something is wrong. This is manual but tells you a lot fast.

---

**Method 2: Visualization (t-SNE or UMAP)**

You can take all 70 vectors (each with 384 numbers) and use a mathematical technique
called **t-SNE** or **UMAP** to compress them down to just 2 numbers — (x, y) coordinates.

This loses most of the information, but it lets you plot all 70 restaurants on a 2D
chart and see if they cluster by type:

```
Rough expected layout:

             (upscale, romantic)
         Mama's ● ● The Beach House
                     ● Ferraro's

(local, cheap)                       (japanese, fresh)
  Tin Roof ●              Sansei ● ● Ama Sushi
  Helena's ●                    ● Nobu

             (casual, beach, shrimp)
               Giovanni's ● ● Paia Fish
```

If the chart looks like this — grouped by cuisine and atmosphere — the vectors are
capturing meaning correctly. If the dots are scattered randomly, the system needs work.

We haven't built this visualization for Ka'ana yet, but it is a standard tool used by
every AI team to validate their embeddings.

---

**Method 3: Blind test with a domain expert**

Create a set of 20 test profiles. For each profile, have someone who knows Hawaii
restaurants well (Todd, a local guide, a hotel concierge) independently list the top 5
restaurants they would recommend. Then run Ka'ana with the same profile and compare.

```
Test profile: vegan, solo explorer, healthy light, scenic, Kauai, budget $$
Human expert's picks: Nourish Kauai, Kauai Juice Co, The Right Slice, Bar Acuda
Ka'ana's picks:       Nourish Kauai (#1), Kauai Juice Co (#3), Bar Acuda (#7)...

Overlap = 3 out of 4 top picks → 75% agreement → good signal ✓
```

The higher the overlap between human expert picks and Ka'ana's top results, the better
the system is working. This is the most meaningful validation you can do.

---

**Method 4: Compare against a simpler system**

Build a much simpler version of the same task — for example, a pure keyword/filter
approach — and compare the two systems on the same test profiles. If Ka'ana's output
is meaningfully better (in the human expert's judgment), the embedding approach is
justified. If they perform the same, the simpler system might be preferred because it
is more transparent.

This comparison is called an **ablation study** or **baseline comparison**. It is how
every serious AI system is evaluated in industry.

---

**Summary: The 384 numbers are a black box, but the system's behavior is fully testable.**

You cannot open the vector and read it. But you can watch the system make thousands of
decisions and check whether those decisions make sense. That is how every AI system in
the world is validated — not by reading its internal numbers, but by measuring its
outputs.

---

#### Question B — What if the user gives very little input? Does the system break?

Short answer: **No. The system always produces exactly 384 numbers regardless of how
much or how little input is given.** It never breaks. But the recommendations become
less specific — which is exactly the right behavior.

Here is what actually happens.

**SBERT always outputs 384 numbers, no matter what.**

It doesn't matter if you give it one word or five paragraphs. The model always processes
the input and produces exactly 384 numbers. That is how the architecture is built.

So even if a user only selects one thing — say, "Maui" — the system still works:

```
User selects only island = "maui", everything else stays at default

UserEncoder builds text:
  "I am looking for a restaurant.
   any cuisine all proteins meat fish vegetarian everything.   ← dietary default
   group of friends social loud fun lively drinks sharing.    ← occasion default
   traditional Hawaiian local plate lunch poke kalua pig.     ← cuisine default
   beachfront outdoor oceanfront toes in sand tiki casual.    ← atmosphere default
   Budget: moderate mid-range casual dining twenty to forty.  ← budget default (2)
   Noise: quiet conversation easy talk relaxed moderate.      ← noise default (2)
   Island: Maui Paia Kihei Wailea Kapalua Hana upcountry."   ← the one thing user picked
```

See what happened? The system filled in all the other fields with their **default values**.
The defaults are the most common/neutral choices:

```
dietary      → "no_restrictions"   (eat anything)
occasion     → "friends_group"     (casual social)
cuisine      → "local_hawaiian"    (most requested in Hawaii)
atmosphere   → "beach_outdoor"     (most requested in Hawaii)
budget       → 2                   (moderate $$)
noise_pref   → 2                   (relaxed)
```

So "minimal input" does not mean "no vector" — it means "vector built mostly from
defaults". The recommendation you get is essentially: "broadly popular restaurants on
Maui that fit the most common preferences". That is still a useful and valid answer.

**The vector for minimal input sits near the "center" of the space**

Here is a way to picture this. On our globe (the unit sphere), restaurants with
very specific personalities are pins near the poles and the edges — unique locations.
Restaurants that are broadly popular and appeal to many people sit somewhere in the
middle — no extreme position.

A user who gives very little input produces a vector that also sits near the middle —
it does not strongly pull toward any specific corner of the globe. The nearest
restaurants will be the broadly popular, generalist places.

A user who fills in every field precisely produces a vector that sits at a very specific
location. The nearest restaurants will be the ones that are a very good fit for exactly
that personality.

```
Minimal input:
  User vector → sits near "center" → matches broadly popular restaurants
  → Results: "crowd pleasers, broadly appealing, most popular" restaurants

Full detailed input:
  User vector → sits at a specific point → matches very specific restaurants
  → Results: "exactly this kind of person's ideal restaurant"
```

Both are valid. The system degrades gracefully — the less you tell it, the more generic
the recommendations, but never wrong or broken.

**What if a user leaves out one dimension?**

In Ka'ana's design, you cannot actually leave things empty — every field has a default
value that gets used if not changed. But if somehow a field were missing, SBERT would
simply produce a vector based on the remaining text, and the missing dimension would be
treated as if it had its default value. The system would continue working.

**The real minimum viable input**

In practice, the two fields that matter most to override are:
1. **Island** — because all the other defaults produce a reasonable "tourist" profile,
   but island determines which 22/22/14/12 restaurants you are searching within.
2. **Occasion** — because "romantic date" vs "family with kids" changes the recommendation
   dramatically more than any other single field.

If a user only picks these two and leaves everything else as defaults, the recommendations
are already quite personalized and useful.

---

### Step 5 — Hard filter: dietary restrictions (`recommender.py`)

Now we apply the first "hard filter". A hard filter is a non-negotiable rule. If a
restaurant fails it, the restaurant is removed completely, regardless of how good its
cosine similarity score was.

The dietary filter works like this:

```
User selected "pescatarian"?
→ The restaurant MUST have "seafood_heavy" in its dietary_tags
→ Any restaurant without this tag is removed

User selected "vegan"?
→ The restaurant MUST have "vegan_options" in its dietary_tags
→ Any restaurant without this tag is removed

User selected "gluten_free"?
→ The restaurant MUST have "gluten_free_options" in its dietary_tags
→ Any restaurant without this tag is removed

User selected "no_restrictions"?
→ No filter applied — all restaurants pass
```

Safety rule: if all restaurants get removed (unlikely but possible), we restore the
full list rather than showing zero results.

---

### Step 6 — Hard filter: price and open hours (`vector_store.py`)

A second round of hard filtering:

**Price filter:**
```
User's budget = 3 (= $$$, upscale)
→ Remove any restaurant with price_range = 4 (fine dining = $$$$)
→ Fine dining is over the budget — remove it regardless of semantic match
```

**Open hours filter:**
```
User's current_time = 19 (7:00 PM)
→ For each restaurant, check if 7 PM falls within open_hours
→ If restaurant closes at 6 PM, it is removed
→ If restaurant is open until 10 PM, it stays
```

Note: some restaurants wrap past midnight. For example, `open: 22, close: 2` (10 PM to
2 AM). The code handles this correctly.

Safety rule: if filtering removes everything, restore the previous (dietary-filtered) list.

---

### Step 7 — Score every remaining restaurant (`recommender.py`)

After filtering, we have a pool of restaurants that passed all the hard rules. Now we
calculate a **final score** for each one. This score determines the ranking.

The final score blends two components:

```
final_score = (cosine_score × 0.65) + (attr_score × 0.35)
```

Let's break down each component.

**Component 1: Cosine Score (65% of final score)**

This is the semantic similarity we calculated in Step 4 — how closely the restaurant's
384-number vector matches the user's 384-number vector. It captures nuanced meaning:
vibe, cuisine style, atmosphere, all rolled into one number.

```
cosine_score ranges from 0.0 to 1.0
0.50 = decent match
0.70 = strong match
0.85 = exceptional match (almost never hit exactly)
```

**Component 2: Attribute Score (35% of final score)**

This scores concrete, practical compatibility across 5 sub-checks:

```
Price fits budget?              → +0.30 (most important practical check)
Restaurant is open right now?   → +0.20
Occasion tags match?            → +0.20 (e.g., "romantic" ↔ "romantic_date")
Noise level within ±1?          → +0.15 (e.g., user wants 2, restaurant is 3 → ok)
Restaurant on user's island?    → +0.15

Maximum attr_score = 1.00
```

Each sub-check adds its points if the condition is met, 0 if not. The attr_score is
the sum of all points earned.

**Why 65/35 split?**

The semantic score captures the "soul" of the match — would this restaurant delight
this person? The attribute score captures practical constraints — can they even go there?

65% semantic ensures that a restaurant that is a truly perfect vibe match gets ranked
highly even if it scores imperfectly on some attributes. 35% attribute ensures that
practical reality (budget, hours) is meaningfully factored in, not ignored.

---

### Step 8 — Generate plain-English explanations (`recommender.py`)

After scoring, for each restaurant we generate a list of human-readable reasons why
Ka'ana picked it. This is done with simple if/then logic — not AI.

Examples:

```python
if user.dietary == "pescatarian" AND "seafood_heavy" in restaurant.dietary_tags:
    → "Seafood-forward menu fits your pescatarian preference"

if user.occasion == "romantic_date" AND "romantic" in restaurant.occasion_tags:
    → "Ideal setting for a romantic dinner"

if user.atmosphere == "beach_outdoor" AND "beachfront" in restaurant.tags:
    → "The beach and ocean setting you're looking for"

if cosine_score > 0.80:
    → "Exceptional taste match (83% similarity)"
elif cosine_score > 0.65:
    → "Strong taste match (71% similarity)"

if restaurant is currently open:
    → "Open right now"
```

These reasons are stored alongside the score and returned to the browser.

---

### Step 9 — Sort and return the top 10 (`recommender.py`)

All scored restaurants get sorted from highest `final_score` to lowest. The top 10 are
packaged into a response and sent back to the browser.

The response looks like this:

```json
{
  "count": 10,
  "recommendations": [
    {
      "restaurant": { "name": "Mama's Fish House", "island": "maui", ... },
      "final_score": 0.7891,
      "cosine_score": 0.8234,
      "attr_score": 0.7500,
      "explanation": [
        "Seafood-forward menu fits your pescatarian preference",
        "Ideal setting for a romantic dinner",
        "Upscale and elegant atmosphere",
        "Strong taste match (82% similarity)",
        "Open right now"
      ],
      "match_breakdown": {
        "semantic_similarity": 0.8234,
        "price_match": 1.0,
        "occasion_match": 1.0,
        "noise_match": 1.0,
        "open_now": 1.0,
        "island_match": 1.0,
        "overall_score": 0.7891
      }
    },
    ...9 more
  ]
}
```

---

### Step 10 — The browser displays the results (`static/index.html`)

The browser receives the JSON and uses JavaScript to build the visual cards you see.
For each recommendation it creates:

- Restaurant name and rank (#1, #2, etc.)
- Island badge, price, open/closed tag
- A score percentage (e.g., "79% match")
- A horizontal bar showing the 65%/35% split between semantic and attribute scores
- The list of plain-English reasons
- The full restaurant description
- (For top 3) A comparison table showing how each dimension scored

All of this is built dynamically in the browser — the server only sent raw data, the
browser built the visual display.

---

## Part 11 — The Files and What Each One Does

Here is every file in the project and its exact role:

```
seed_data.py       → The raw data. 70 restaurant dictionaries. Pure content,
                     no logic. This is like the database of restaurants.

models.py          → The encoders. Contains RestaurantEncoder and UserEncoder.
                     Both use SBERT to convert text to 384-D vectors.
                     Also contains all the translation tables (DIETARY_MAP,
                     OCCASION_MAP, CUISINE_MAP, etc.)

seed_runner.py     → The startup script. Runs once when the server starts.
                     Encodes all 70 restaurants and loads them into Qdrant.

vector_store.py    → The Qdrant wrapper. Handles all interaction with the
                     vector database: storing vectors, querying by similarity,
                     applying price and hours filters.

recommender.py     → The brain. Takes the user profile and does: encode user →
                     query Qdrant → dietary filter → score → explain → top 10.

main.py            → The front door. FastAPI server that receives HTTP requests
                     from the browser, validates them, calls recommender.py,
                     and sends results back.

static/index.html  → The UI. Everything the user sees. Sends requests to the
                     server, receives results, builds the visual cards.
```

---

## Part 12 — A Visual Map of the Entire System

```
═══════════════════════════════════════════════════════════════════════
  SERVER STARTUP — happens once, takes ~5 seconds
═══════════════════════════════════════════════════════════════════════

  seed_data.py                   models.py             vector_store.py
  ┌──────────────┐               ┌──────────────────┐  ┌─────────────┐
  │ 70 restaurant│               │ RestaurantEncoder │  │             │
  │ dictionaries │──────────────▶│                  │  │   QDRANT    │
  │              │               │ 1. dict → text   │  │   (RAM)     │
  │ {name, tags, │               │ 2. SBERT → 384-D │─▶│             │
  │  island,     │               │ 3. normalize     │  │ 70 vectors  │
  │  dietary,    │               └──────────────────┘  │ 70 payloads │
  │  occasion,   │                                     │             │
  │  price...}   │                                     └─────────────┘
  └──────────────┘

═══════════════════════════════════════════════════════════════════════
  PER REQUEST — happens every time you click the button (~30ms)
═══════════════════════════════════════════════════════════════════════

  Browser                main.py          models.py
  ┌──────────────────┐   ┌───────────┐   ┌──────────────────┐
  │ Taste ID         │   │           │   │ UserEncoder      │
  │ dietary          │──▶│  FastAPI  │──▶│                  │
  │ occasion         │   │  receives │   │ 1. answers→text  │
  │ cuisine          │   │  validates│   │ 2. SBERT → 384-D │
  │ atmosphere       │   └───────────┘   │ 3. normalize     │
  │ budget           │                   └────────┬─────────┘
  │ noise / time     │                            │ user vector
  │ island           │                            ▼
  └──────────────────┘            vector_store.py
         ▲                        ┌──────────────────────────┐
         │                        │ Qdrant cosine search     │
         │                        │ vs all 70 restaurants    │
         │                        │ → top 50 by similarity   │
         │                        └─────────────┬────────────┘
         │                                      │ 50 candidates
         │                                      ▼
         │                        recommender.py
         │                        ┌──────────────────────────┐
         │                        │ Hard filter: dietary     │
         │                        │ Hard filter: price+hours │
         │                        │                          │
         │                        │ For each remaining:      │
         │                        │  score = 0.65×cosine     │
         │                        │         + 0.35×attr      │
         │                        │  generate explanations   │
         │                        │                          │
         │                        │ Sort by score            │
         │                        │ Return top 10            │
         │                        └─────────────┬────────────┘
         │                                      │ 10 results (JSON)
         │                                      ▼
         └──────────────── browser renders cards ◀─────────────
```

---

## Part 13 — Common Questions

### "Is this real AI?"

Yes and no. The SBERT model that converts text to vectors is a real neural network with
millions of learned parameters — that part is genuine AI. But the scoring formula
(65%/35%), the translation tables, the filters, and the explanation generator are all
hand-written logic by a human. The system is a hybrid: AI-powered search + rule-based
scoring + rule-based explanation.

### "Are the 70 restaurants real?"

Yes. Every restaurant in the dataset is a real, operating restaurant in Hawaii as of
the time this system was built. Their locations, cuisine styles, and descriptions are
based on real information. Opening hours and prices are approximate and may have changed.

### "How accurate are the recommendations?"

For a prototype with 70 restaurants and no real user data, the recommendations are
surprisingly good. They correctly surface thematic matches — if you ask for vegan,
healthy, scenic options on Kauai, that's what you get. The main limitation is that the
system has never learned from real user behavior (clicks, orders, ratings), so it
cannot personalize beyond the Taste ID inputs.

### "What would make it better?"

1. **Real user signals**: track which restaurants users actually visit. Use that to
   refine the model weights over time.
2. **More restaurants**: 70 is a prototype. A real product would have thousands.
3. **Real-time data**: pricing, hours, and availability could be fetched live.
4. **User history**: learn each user's preferences over multiple visits.
5. **A/B testing**: try different scoring weights (is 65/35 the best split?) and
   measure which version users like more.

### "Why not just use ChatGPT to recommend restaurants?"

You could ask ChatGPT "what are good romantic pescatarian restaurants on Maui?" and
get decent results. But:

1. ChatGPT's knowledge is static (a training cutoff date) — it won't know about new
   restaurants or updated hours.
2. It cannot be given a structured database of 70 specific restaurants and asked to
   rank them against a Taste ID in real time.
3. It cannot learn from user behavior.
4. It would be far more expensive and slower than our system for a production app.

Ka'ana's approach gives you the semantic understanding of AI (via SBERT) combined with
the precision of a structured database and scoring system.

---

## Part 14 — Glossary: Every Technical Term, Explained Simply

| Term | What it actually means |
|------|------------------------|
| **Vector** | A list of numbers. [0.12, -0.34, 0.71, ...] — that's a vector. |
| **Embedding** | Same as vector. The word "embedding" emphasizes that it was produced by an AI to represent meaning. |
| **384-D** | The list has 384 numbers. "D" stands for dimensions. |
| **SBERT** | The AI model (`all-MiniLM-L6-v2`) that reads text and produces a 384-number vector. |
| **Encoder** | Anything that converts input (text, images, etc.) into a vector. |
| **Two-tower** | Two encoders (one for restaurants, one for users) that share the same underlying AI model so their outputs can be compared. |
| **Normalize / L2 norm** | Rescale a vector so its total size becomes exactly 1.0. |
| **Unit sphere** | The imaginary globe where all normalized vectors live (radius = 1.0). |
| **Cosine similarity** | A number (0.0 to 1.0) measuring how similar two vectors are, based on the angle between them. |
| **Dot product** | Multiply each pair of numbers in two vectors and sum the results. For normalized vectors, this equals cosine similarity. |
| **Qdrant** | A vector database — stores vectors and can search by similarity very quickly. |
| **`:memory:`** | A Qdrant mode that keeps data in RAM instead of saving to disk. Fast, but lost on restart. |
| **Payload** | The full restaurant data stored alongside a vector in Qdrant. |
| **Upsert** | "Update or insert" — store something, replacing it if it already exists. |
| **Seed / Seeding** | Encoding all 70 restaurants and loading them into Qdrant at startup. |
| **Hard filter** | A non-negotiable rule that removes restaurants regardless of their score. |
| **Soft filter** | A preference that affects score rather than eliminating restaurants outright. |
| **Late fusion** | Combining two scores (semantic + attribute) at the end of the pipeline. |
| **Taste ID** | Ka'ana's 5-dimension user preference profile: dietary + occasion + cuisine + atmosphere + budget. |
| **Pareto principle** | The idea that 20% of the right inputs drive 80% of the results. Ka'ana uses 5 key dimensions to capture 80%+ of recommendation quality. |
| **JSON** | JavaScript Object Notation — a standard format for sending data between a browser and a server. Looks like: `{"key": "value"}`. |
| **FastAPI** | The Python web framework that runs the server, receives requests from the browser, and sends back responses. |
| **POST request** | The way a browser sends data *to* a server (as opposed to GET, which is how a browser *receives* a page). |
| **dietary_tags** | Tags on each restaurant marking dietary compatibility (e.g., `"vegan_options"`, `"gluten_free_options"`). |
| **occasion_tags** | Tags on each restaurant marking best use cases (e.g., `"romantic"`, `"family"`, `"solo"`). |

---

## Part 15 — Honest Comparison: This POC vs. Todd's Current Ka'ana System

This section compares the system built in this document against the system Todd
described in his email. The goal is to be completely factual — not to oversell either
approach. Both have real strengths and real weaknesses.

First, a summary of what Todd described:

> *"We currently get onboarding data regarding cuisine types and dietary preferences.*
> *That sets a very rudimentary taste preferences profile (basic Taste ID). Then Ana has*
> *a detailed set of preferences to further build out the Taste ID. Touch points are*
> *stacked based on reels watched, restaurant profiles viewed, items clicked, menu items*
> *clicked, directions requested, Ana interactions, back-and-forth dialogue. We are also*
> *developing a gamified method to acquire points that verifies places visited and foods*
> *ordered."*

That is Todd's current system. Let's compare it honestly against our POC.

---

### Where the two systems are fundamentally different

Before scoring anything, it is important to understand that these two systems are
**solving different parts of the same problem**. They are not direct competitors —
they are complementary.

- **Todd's system** is designed for users who have been using Ka'ana for weeks or months.
  It gets better the more you use it. It learns from your behavior.

- **Our system** is designed for users who just opened the app for the first time.
  It gives a strong, personalized recommendation immediately, without any behavioral history.

In the industry, these two challenges have names:
- The problem our system solves is called the **cold start problem** — how to help a
  brand new user before you know anything about them.
- The problem Todd's system solves is called **long-term personalization** — how to get
  better and better at serving someone as they keep using your product.

A great production recommendation system needs both. Keep that in mind as you read
the comparison below.

---

### Side-by-side comparison

| Dimension | Todd's Current System | Our SBERT POC |
|-----------|----------------------|---------------|
| **Initial onboarding dimensions** | 2 (dietary + cuisine) | 7 (dietary, occasion, cuisine, atmosphere, budget, noise, island) |
| **Matching method** | Not specified — likely rule-based or category matching | Semantic vector similarity (SBERT 384-D embeddings) |
| **Understands meaning vs. keywords** | Unknown — depends on Ana's implementation | Yes — "romantic" and "intimate" are treated as equivalent concepts |
| **Cold start (day 1 user)** | Rudimentary — Todd himself calls it "very rudimentary" | Strong — personalized from the first click |
| **Long-term personalization** | Yes — grows richer with every interaction | No — POC gives same result for same input every time |
| **Behavioral signals** | Yes — clicks, views, reels, directions, Ana dialogue | None — this is a POC |
| **Visit verification** | Planned (gamified points system) | None |
| **Explains why a restaurant was picked** | Unknown — not described | Yes — plain-English reasons for every recommendation |
| **Dietary hard filter** | Yes (part of onboarding) | Yes |
| **Price awareness** | Unknown | Yes — hard price ceiling filter |
| **Open hours awareness** | Unknown | Yes — checks if restaurant is open right now |
| **Island-level filtering** | Unknown | Yes — explicit across all 4 islands |
| **Number of restaurants in dataset** | Unknown — likely larger as a real app | 70 (POC only) |
| **Restaurant data quality** | Unknown | 70 real, researched Hawaii restaurants |
| **Production-ready** | Yes (it's a running app) | No (local prototype) |
| **Has real users** | Yes | No |

---

### What Todd's system does better — honestly

**1. Behavioral learning (the biggest advantage)**

Todd's system tracks what users actually do: which restaurants they tap, which reels
they watch, which menu items they click, whether they request directions (a strong signal
they intend to visit). Over time, this builds a picture of real preferences — not stated
preferences.

This is a massive advantage because **stated preferences and actual preferences often
differ**. A person might tell the app they want healthy food, but consistently click on
burger places. Todd's system catches this. Ours never will, because we have no behavioral
data.

This is the single biggest gap in our POC.

**2. Visit verification**

The gamified points system Todd mentions — where users verify that they actually visited
a restaurant — produces extremely valuable ground truth. If you know user A visited
Mama's Fish House and user B visited Da Kitchen, you have real signal about their
revealed preferences. Our system has no such data.

**3. Conversation with Ana**

If Ana is powered by a capable language model and conducts genuine back-and-forth
dialogue with users about food preferences, it can capture nuance that our 5 radio
buttons cannot. For example, Ana could ask "Did you enjoy the restaurant you tried last
week?" and update the Taste ID accordingly. Our onboarding is one-direction only.

**4. Community and social layer**

Todd mentions the Taste ID becoming a "user profile in the Ka'ana community". Social
signals (what people with similar Taste IDs like) are powerful. Our system has no
community dimension at all.

---

### What our system does better — honestly

**1. Cold start: day 1 recommendations**

Todd explicitly says his system starts "very rudimentary". A new user gets a generic
recommendation. Our system gives a semantically personalized recommendation from the
first tap — even with zero behavioral history. For Ka'ana, which is trying to attract
and retain new users, this gap matters a great deal. A weak first experience loses users
before the behavioral learning has any chance to kick in.

**2. Semantic understanding vs. category matching**

This is a structural difference. If Todd's matching is rule-based or category-based
(which is the most common approach and likely what "rudimentary" implies), it matches
on exact categories. A restaurant tagged "seafood" matches a user who picked "seafood".
A restaurant tagged "Japanese" matches a user who picked "Japanese".

Our system matches on meaning. A user who wants "light, fresh, clean food" will be
matched against restaurants described as "organic, health-conscious, farm-to-table" even
if neither the user nor the restaurant used the same words. This semantic bridge is not
possible with keyword or category matching.

To be fair: we do not know exactly how Ana enriches the Taste ID. If Ana uses a capable
language model to interpret restaurant descriptions, it might close this gap. But Todd's
email does not describe that, so we cannot assume it.

**3. Richer onboarding dimensions**

Our system captures 7 dimensions upfront: dietary, occasion, cuisine, atmosphere,
budget, noise preference, and island. Todd's system starts with 2: dietary and cuisine.

More dimensions upfront = better day-1 recommendations. This is especially valuable
in Hawaii's tourism context, where most users are visitors who will only use the app
once or twice — they will never accumulate behavioral history. For a tourist who opens
Ka'ana once during their trip, the entire value of the app rests on the quality of that
first recommendation.

**4. Transparent, structured scoring**

Our system's scoring formula is explicit and inspectable:
```
final_score = (cosine_score × 0.65) + (attr_score × 0.35)
```
Every component is known, adjustable, and auditable. Todd's system does not describe
its scoring formula. If you cannot explain why restaurant A ranked above restaurant B,
that is a problem when presenting to investors or partners.

**5. Explainability**

Our system tells users exactly why each restaurant was recommended in plain English.
Todd's system does not describe any explainability feature. Explainability builds trust
and helps users refine their preferences ("oh, it picked this because it thought I
wanted beach dining — I actually meant indoor").

**6. Open hours and price ceiling enforcement**

Our system checks whether each restaurant is currently open at the user's stated time,
and removes restaurants that exceed the user's budget as a hard rule. Todd's system does
not mention either of these. Recommending a closed restaurant is an immediate trust
failure.

---

### The honest verdict

Neither system is categorically better. They are strong in different areas.

| | Todd's System | Our POC |
|--|---|---|
| **Better for** | Returning users with behavioral history | New users on their first session |
| **Biggest strength** | Learns from real behavior over time | Semantic understanding + strong cold start |
| **Biggest weakness** | Weak day-1 experience (own words: "rudimentary") | No behavioral learning at all |
| **Production status** | Running app with real users | Local prototype, 70 restaurants |

**The most important thing to communicate to Todd is not "ours is better". It is this:**

> Ka'ana's growth depends on new users having a great first experience.
> Behavioral signals only kick in after repeat usage — but most Hawaii visitors
> will only use the app once or twice.
> The semantic approach solves the cold start problem that Todd's current system
> leaves unaddressed.
> The ideal Ka'ana system combines both: semantic SBERT retrieval for day-1 users,
> then behavioral signals layered on top as users engage more.

This is a known pattern in production recommendation systems. Spotify, Netflix, and
Airbnb all use exactly this hybrid: semantic/content-based retrieval for new users,
collaborative filtering (behavioral) for experienced users.

---

### What the combined system would look like

If both approaches were merged into one production system:

```
NEW USER (day 1, no history)
  → 5-dimension Taste ID onboarding (our approach)
  → SBERT semantic retrieval from restaurant vectors
  → Strong day-1 recommendation, fully personalized
  → User trust established from first session

RETURNING USER (days 2–100, behavioral history building)
  → Taste ID refined by: clicks, views, reels watched, Ana dialogue
  → Behavioral signals layer on top of semantic vectors
  → Recommendations improve with every interaction
  → Visit verification adds ground truth

LONG-TERM USER (months of history)
  → Dense behavioral profile
  → Can surface restaurants they never would have found through onboarding
  → Community signals: "people with your Taste ID also loved..."
```

This is the system worth building. Our POC demonstrates that the semantic layer is
technically viable and produces meaningful results today, without any behavioral data.
Todd's system demonstrates that the behavioral layer is operationally feasible.
The missing piece is combining them.

---

### One caveat about this comparison

We do not have full visibility into Todd's system. We know what he described in his
email. It is possible that:

- Ana uses a sophisticated AI that closes the semantic gap — we cannot verify this
- Their restaurant database is far larger and better curated than our 70
- Their matching is more sophisticated than the email implies

Any of these would change the comparison. This document is based strictly on what
Todd described. Present it as a discussion starter, not as a final judgment.

---

*Last updated: April 2026*
*This document covers the Ka'ana POC — 70 restaurants, 4 islands, SBERT-based two-tower retrieval.*
