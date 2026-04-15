# What Questions Should Ka'ana Ask? — Research-Backed Taste ID Design
### Finding the minimum questions that capture maximum preference signal

---

## Why This Document Exists

Todd's core challenge, stated plainly:

> *"I have been interested in the 20 percent of taste ID details that give 80+ percent
> accuracy in matching."*

This document answers that question using published academic research — not opinion.
It identifies exactly which dimensions drive restaurant choice, in what order of
importance, and proposes a methodology for capturing them with the fewest possible
questions. Every claim is sourced.

---

## Part 1 — What Does the Research Say About Why People Choose Restaurants?

Over the past 30 years, researchers have studied restaurant choice behavior extensively.
The most rigorous studies use a technique called **conjoint analysis** — where participants
are given pairs or sets of hypothetical restaurants and asked to choose, revealing which
attributes they weight most heavily through their actual choices rather than what they
*say* is important.

Here is what the research consistently finds.

### The "Big Four" restaurant attributes

Virtually every large-scale study converges on the same four top-level categories:

| Rank | Attribute | What it covers |
|------|-----------|---------------|
| 1 | **Food** | Cuisine type, food quality, taste, menu variety, freshness |
| 2 | **Price & Value** | How expensive, whether it feels worth it |
| 3 | **Atmosphere** | Setting, noise level, décor, indoor/outdoor, vibe |
| 4 | **Service** | Staff quality, wait times, attentiveness |

> **Source:** Chua, B.L., Karim, S., et al. (2020). "Customer Restaurant Choice: An
> Empirical Analysis of Restaurant Types and Eating-Out Occasions."
> *International Journal of Environmental Research and Public Health*, 17(17), 6276.
> [Full text (PMC, open access)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7503372/)

> **Source:** Exploring the comparative salience of restaurant attributes: A conjoint
> analysis approach. *International Journal of Information Management*, 36(6), 2016.
> [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0268401216301189)

### Specific importance weights from conjoint analysis

When researchers used conjoint analysis on restaurant attributes, the weights came out:

```
Food quality:    25.2%  ← most important by a clear margin
Price:           16.4%
Service quality: 15.0%
Atmosphere:      13.1%
Location:        ~10%
Other:           ~20%
```

> **Source:** Choice-based conjoint analysis study on restaurant attributes,
> *International Hospitality Review*, 2022.
> [Emerald Insight](https://www.emerald.com/insight/content/doi/10.1108/ihr-12-2022-0059/full/html)

### The single most important modifier: Occasion

Here is the critical insight that most recommendation systems miss:

**The same person will choose a completely different restaurant depending on the
occasion.** The occasion does not just influence preference — it reorganizes the
entire ranking of what matters.

From the research:

```
Quick meal / convenience:  Price matters most. Location matters. Ambiance irrelevant.
Social outing with friends: Atmosphere matters most. Price secondary.
Business necessity:         Brand reputation matters most. Food quality secondary.
Celebration / special:      Word-of-mouth and ambiance matter most. Price least.
Romantic date:              Atmosphere first. Food second. Price barely registers.
```

> **Source:** Kim, Y.H. & Chung, J.Y. "Restaurant Selection Criteria: Understanding
> the Roles of Restaurant Type and Customers' Sociodemographic Characteristics."
> [Semantic Scholar](https://www.semanticscholar.org/paper/Restaurant-Selection-Criteria:-Understanding-the-of-Kim-Chung/e54d0bd00cf6e3ad68adbb058f3e23767339841f)

> **Source:** Chua et al. (2020), PMC — eating-out occasion as key determinant of
> selection criteria.
> [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7503372/)

**Implication for Ka'ana:** Occasion is not just one of many inputs — it is the
*master variable* that determines what all other answers mean. It must be the first
or second question asked. Without it, all other answers are ambiguous.

### Food adventurousness: the hidden personality trait

Beyond the obvious attributes, research has identified a deep personality trait that
predicts restaurant behavior remarkably well: **food neophobia vs. food neophilia**.

- A **food neophobe** avoids unfamiliar foods. They reliably choose familiar cuisines,
  stick to what they know, avoid ethnic restaurants they haven't tried, and feel
  discomfort when the menu has nothing familiar on it.
- A **food neophile** actively seeks new and unfamiliar food experiences. They are the
  person who wants to try the chef's tasting menu at the newest restaurant in town.

This trait is stable across a person's lifetime (it is a personality trait, not a mood).
It was formally measured by Pliner & Hobden's **Food Neophobia Scale (FNS)** in 1992 —
10 questions with validated test-retest reliability, translated into 15+ languages.

> **Source:** Pliner, P. & Hobden, K. (1992). "Development of a scale to measure the
> trait of food neophobia in humans." *Appetite*, 19(2), 105–120.
> [PubMed](https://pubmed.ncbi.nlm.nih.gov/1489209/) |
> [ResearchGate PDF](https://www.researchgate.net/publication/21666930_Development_of_a_scale_to_measure_the_trait_of_food_neophobia_in_humans)

**Why this matters for Ka'ana:** A single well-designed question capturing adventurousness
(without using the word "adventurous") is one of the highest-value questions you can ask.
It predicts an enormous range of downstream behavior — which cuisine types will appeal,
whether a fusion restaurant will excite or alienate, whether "unusual" is a selling point
or a warning. It is the dimension most existing restaurant apps completely ignore.

---

## Part 2 — What the Research Reveals About Asking Fewer Questions

### The problem with asking too many questions

Research on user profiling is clear on one thing: users abandon questionnaires.
The longer the onboarding, the higher the drop-off rate. Explicit user profiling
(asking people to fill forms) "quickly became problematic as users were reluctant to
disclose information or found the form-filling process cumbersome."

> **Source:** User Modeling and User Profiling: A Comprehensive Survey, arXiv 2402.09660
> [arXiv](https://arxiv.org/html/2402.09660v2)

The research field that solves this is called **preference elicitation** — the science
of choosing which questions to ask, in which order, to maximize what you learn about
a person with the minimum number of interactions.

### The active learning approach

The most rigorous method for minimizing questions is **active learning**. The idea:
instead of asking a fixed questionnaire to every user, choose the next question based
on what you've already learned from previous answers, picking whichever question
eliminates the most uncertainty.

Think of it like a doctor doing a diagnosis. They do not ask every possible question.
They ask "any chest pain?" and if yes, follow up with "when does it happen?" If no,
they go a different direction entirely. Each answer narrows the space.

Applied to restaurants: if someone says they want a quiet, romantic dinner, you do not
need to ask if they want a loud bar. You have already eliminated that entire half of
the option space. The next question should be about something you still do not know —
cuisine preference within the upscale/quiet/romantic segment.

> **Source:** "Explainable Active Learning for Preference Elicitation" (2023),
> arXiv:2309.00356.
> [arXiv](https://arxiv.org/abs/2309.00356)

> **Source:** "Improving preference elicitation in a conversational recommender system
> with active learning strategies."
> [ResearchGate](https://www.researchgate.net/publication/351150845_Improving_preference_elicitation_in_a_conversational_recommender_system_with_active_learning_strategies)

> **Source:** "Active learning algorithm for alleviating the user cold start problem
> of recommender systems" (2025). *Scientific Reports*.
> [Nature](https://www.nature.com/articles/s41598-025-09708-2)

### The pairwise comparison approach

A second well-researched method is **pairwise preference elicitation**: instead of
asking "what do you want?" (which requires introspection), you show two options and ask
"which of these two would you rather go to tonight?"

This is cognitively much easier and produces more reliable data, because it does not
require the user to know or articulate their preferences — it just requires them to react
to concrete choices. You reveal preferences through revealed behavior, not stated opinion.

> **Source:** "Personalized Recommendations via Active Utility-based Pairwise Sampling"
> (2025), arXiv:2508.14911.
> [arXiv](https://arxiv.org/html/2508.14911)

> **Source:** "Active Preference Elicitation via Adjustable Robust Optimization."
> [UMD PDF](http://www.cs.umd.edu/~dmcelfre/files/Active_Preference_Elicitation_via_Adjustable_Robust_Optimization.pdf)

### The conversational LLM approach (newest, 2024-2025)

The most recent research uses large language models (LLMs like GPT-4) to conduct a
natural conversation that extracts preferences. Instead of a form, the user talks.
The LLM asks clarifying questions, picks up on implicit signals ("I visited Maui last
year and loved the fish shacks"), and builds a preference profile without the user
realizing they're being profiled.

> **Source:** "Asking Clarifying Questions for Preference Elicitation With Large Language
> Models" (2025), arXiv:2510.12015.
> [arXiv](https://arxiv.org/abs/2510.12015) |
> [Google Research](https://research.google/pubs/asking-clarifying-questions-for-preference-elicitation-with-large-language-models/)

This is directly relevant to Ka'ana's Ana — if Ana is an LLM-powered conversational
agent, it is already positioned to do this. The question is whether it is being asked
the right things.

---

## Part 3 — The Minimum Viable Questions: Research-Backed Answer

Based on synthesizing all the research above, here is the evidence-based answer to
Todd's question: **what are the minimum questions that capture the most information?**

### The 5 non-negotiable questions (the research-backed Pareto 20%)

Every one of these is supported by multiple studies as a high-signal, irreducible
preference dimension.

---

**Question 1 — Occasion (master modifier)**

> *"What is this dinner for?"*

Why it is #1: Research shows occasion reorganizes the importance ranking of every other
dimension. It is the single highest-information question you can ask. All other answers
are interpreted differently depending on this answer.

Choices: Romantic date / Friends night out / Family & kids / Quick solo meal /
Special celebration / Business / Adventure & discovery

Research backing: Chua et al. (2020), Kim & Chung — both show occasion as the primary
stratifying variable in restaurant choice behavior.

---

**Question 2 — Food adventurousness (personality trait)**

> *"When you travel somewhere new, what sounds more exciting: finding the most authentic
> local food spot the tourists don't know about, or going to a well-known restaurant
> that you know will be great?"*

Why it is #2: This captures the food neophobia/neophilia dimension — a stable personality
trait that predicts a wide range of downstream preferences (cuisine type, familiarity
preference, openness to fusion, etc.) from a single answer. No other single question
predicts as much about a person's food personality.

Choices: "Find the hidden local spot" (high adventurousness) → "Go to the trusted
well-known place" (high familiarity preference), with a middle option.

Research backing: Pliner & Hobden (1992) Food Neophobia Scale. Also: food neophobia
and sensation seeking are correlated (Raudenbush & Frank, 1999).

Note: this question is designed so the user does not feel like they are being tested.
It describes a real scenario (traveling) which is exactly Ka'ana's target user context.

---

**Question 3 — Budget / value threshold**

> *"About how much per person are you comfortable spending on dinner tonight?"*

Why it is #3: Price is the #2 driver overall in conjoint studies and the #1 driver for
casual/quick occasions. It is also a hard constraint — no amount of semantic similarity
matters if the restaurant is $40 over budget. It must be captured.

Choices: Under $20 / $20–40 / $40–80 / $80+ (or equivalent visual: $, $$, $$$, $$$$)

Research backing: Multiple conjoint studies (16.4% importance weight), Kim & Chung
(price as top priority for full-service, quick-casual, quick-service).

---

**Question 4 — Atmosphere / setting**

> *"Where do you want to eat?"*

Why it is #4: Atmosphere is the #3 driver overall, but it jumps to #1 for romantic
and social occasions. It is also the dimension most closely tied to the specific
competitive advantage Ka'ana has in Hawaii — beach, outdoor, and scenic settings are
uniquely available here and are a primary draw for tourists.

Choices: Beachfront & outdoor / Upscale & intimate / Lively & social / Cozy & local /
Scenic view

Research backing: Conjoint analysis (13.1% importance weight), Chua et al. (2020)
— atmosphere as deciding factor once choice set is evoked.

---

**Question 5 — Dietary restriction (hard filter)**

> *"Any dietary needs we should know about?"*

Why it is #5: Dietary restriction is not a preference — it is a hard constraint that
can make a restaurant completely unsuitable regardless of all other scores. It must be
captured early to avoid recommending impossible matches.

Choices: No restrictions / Vegetarian / Vegan / Pescatarian (fish, no meat) /
Gluten-free

Research backing: Dietary restrictions as emerging hard filter in recent restaurant
selection literature (Chua et al., 2020 — plant-based options, healthy menus as
growing factors in restaurant choice).

---

### The 2 high-value optional questions (the next 20% of signal)

These two add significant value but are not strictly necessary if you have the five above.

**Optional A — Cuisine curiosity**

> *"What kind of food are you in the mood for?"*

Value: narrows cuisine style quickly. Less valuable than the 5 above because food
adventurousness (Q2) already gives a partial answer, and semantic matching can often
infer cuisine preference from occasion + atmosphere answers.

**Optional B — Group size / social context**

> *"How many people are joining you?"*

Value: solo diners have very different needs (bar seating, fast service, no judgment)
vs. large groups (noise tolerance, shareable plates, reservations). This is not well
captured by occasion alone.

---

### The 2 Ka'ana-specific questions (context, not taste)

These are not really "taste" questions — they are operational context. But they
dramatically improve recommendation quality in the Ka'ana context.

**Context A — Island**

> *"Which island are you on?"*

Self-explanatory. Non-negotiable for a Hawaii-specific app.

**Context B — Time (current time)**

> *"Are you looking for somewhere to go right now, or planning ahead?"*

If right now: use current time to filter open restaurants. If planning: skip the hours
filter. This is more elegant than asking "what time is it?" and produces the same result.

---

## Part 4 — Ranked Summary: Information Value Per Question

Based on the research, here is each question ranked by how much it reduces uncertainty
about a person's ideal restaurant:

```
RANK  QUESTION                      INFO VALUE    RESEARCH BASIS
────  ───────────────────────────   ──────────    ──────────────────────────────────
 1    Occasion                      Very high     Chua et al. (2020), Kim & Chung
                                                  Reorganizes all other dimensions
 2    Food adventurousness          Very high     Pliner & Hobden (1992) FNS
                                                  Stable personality trait, broad signal
 3    Budget / price threshold      High          Multiple conjoint studies (16.4%)
                                                  Hard constraint, can't be inferred
 4    Atmosphere / setting          High          Conjoint studies (13.1%)
                                                  Jumps to #1 for romantic/social
 5    Dietary restriction           High          Hard filter — must be captured
                                                  One wrong match breaks trust
 6    Cuisine type                  Medium        Partially inferred from Q2 + Q4
 7    Group size / social context   Medium        Not well studied; practical value
 8    Island (Ka'ana-specific)      Very high     Operational context, not taste
 9    Current time (Ka'ana-specific)Medium        Operational context, not taste
```

---

## Part 5 — Why This Approach Is Novel Compared to What's Out There

Most restaurant apps (Yelp, Google Maps, TripAdvisor) ask zero taste questions.
They rely entirely on: location, rating, category tag, price range.

Todd's current system starts with dietary + cuisine. That is 2 of the 9 questions above.

The approach outlined in this document is different from existing methods in three ways:

### 1. Occasion is treated as the master variable, not one of many options

No existing consumer restaurant app uses occasion as the primary organizing variable.
The research is unambiguous that it should be. This is a genuine gap in the market.

### 2. Food adventurousness is captured as a personality trait, not a cuisine list

Every existing app asks "what cuisine do you like?" — which gives a shallow, low-signal
answer (people pick the obvious ones). Asking about adventurousness as a personality
trait via a scenario question ("hidden local spot vs. trusted well-known place") is:
- Validated by 30 years of food psychology research (Pliner & Hobden)
- More novel and engaging as a question
- Gives far richer signal than a cuisine dropdown

### 3. The methodology is active / adaptive, not a fixed form

Research supports using the answers to Q1 (occasion) to adaptively choose which
follow-up questions matter most, rather than asking the same 9 questions to every user.

Example:
```
User picks "Quick solo meal"
→ Q3 (budget) becomes most important to ask next
→ Q4 (atmosphere) becomes much less important — skip or ask last
→ Q2 (adventurousness) still highly valuable

User picks "Romantic date"
→ Q4 (atmosphere) becomes most important to ask next
→ Q3 (budget) becomes less important — most people in a romantic context
  are willing to spend more than their default
→ Q5 (dietary) still non-negotiable
```

This adaptive approach — using each answer to prioritize subsequent questions — is
exactly what the active learning research recommends and is not implemented by any
major consumer restaurant app today.

> **Source:** "Preference Elicitation Strategy for Conversational Recommender System,"
> *ACM WSDM 2019*.
> [ACM](https://dl.acm.org/doi/10.1145/3289600.3291604)

> **Source:** "Aspect-based active learning for user preference elicitation," CIRCLE 2020.
> [CEUR-WS PDF](https://ceur-ws.org/Vol-2621/CIRCLE20_16.pdf)

---

## Part 6 — Manus Research Prompt

If you want Manus to find and download the actual research papers, use this prompt
exactly. Manus can take over a browser, so the prompt is written for that context.

---

```
TASK: Research and download academic papers on restaurant preference elicitation
and taste profiling. Follow the steps below exactly.

STEP 1 — Download these specific papers (open access or author PDFs):

1. Chua, B.L., Karim, S., et al. (2020). "Customer Restaurant Choice: An Empirical
   Analysis of Restaurant Types and Eating-Out Occasions."
   URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7503372/
   Action: Go to this URL, download the full PDF, save as "01_Chua2020_Restaurant_Choice.pdf"

2. Pliner, P. & Hobden, K. (1992). "Development of a scale to measure the trait of
   food neophobia in humans." Appetite, 19(2), 105–120.
   URL: https://www.researchgate.net/publication/21666930
   Action: Download the PDF, save as "02_Pliner_Hobden_1992_Food_Neophobia_Scale.pdf"

3. "Explainable Active Learning for Preference Elicitation" (2023).
   URL: https://arxiv.org/pdf/2309.00356
   Action: Download the PDF directly, save as "03_Active_Learning_Preference_Elicitation_2023.pdf"

4. "Asking Clarifying Questions for Preference Elicitation With Large Language
   Models" (2025).
   URL: https://arxiv.org/abs/2510.12015
   Action: Click "Download PDF" on the arxiv page,
   save as "04_LLM_Clarifying_Questions_2025.pdf"

5. "Active learning algorithm for alleviating the user cold start problem of
   recommender systems" (2025). Scientific Reports.
   URL: https://www.nature.com/articles/s41598-025-09708-2
   Action: Download full text PDF, save as "05_Active_Learning_Cold_Start_2025.pdf"

6. "Cold-start Recommendation by Personalized Embedding Region Elicitation" (2024).
   URL: https://arxiv.org/abs/2406.00973
   Action: Download PDF, save as "06_Cold_Start_Embedding_Elicitation_2024.pdf"

7. "Exploring the comparative salience of restaurant attributes: A conjoint analysis
   approach." International Journal of Information Management, 2016.
   URL: https://www.sciencedirect.com/science/article/abs/pii/S0268401216301189
   Action: If paywalled, search Google Scholar for the author name and title + "PDF"
   and find an open-access version. Save as "07_Conjoint_Restaurant_Attributes.pdf"

8. Restaurant Selection Criteria paper by Kim & Chung.
   URL: https://scholarworks.umass.edu/bitstreams/b3b5f8a3-2357-4e96-a3e7-3d4e97256b43/download
   Action: Download directly, save as "08_Kim_Chung_Restaurant_Selection_Criteria.pdf"

STEP 2 — Search Google Scholar for these additional papers:

Search query 1: "preference elicitation recommender systems minimum questions"
  Filter: published 2020–2025
  Action: Download the top 3 most cited results that are open access.
  Save as "09_pref_elicit_1.pdf", "09_pref_elicit_2.pdf", "09_pref_elicit_3.pdf"

Search query 2: "food personality traits restaurant choice"
  Filter: published 2015–2025
  Action: Download the top 2 most cited open-access results.
  Save as "10_food_personality_1.pdf", "10_food_personality_2.pdf"

STEP 3 — Create a summary document

After downloading, create a plain text file called "00_PAPER_INDEX.txt" listing:
- Paper number
- Title
- Authors and year
- One sentence on what this paper contributes to the restaurant preference question
- Filename

STEP 4 — Package everything

Zip all downloaded PDFs and the index file into a single archive called
"Kaana_TasteID_Research.zip" and place it on the Desktop.

IMPORTANT NOTES:
- If any paper is behind a paywall, do not purchase it. Instead:
  a. Try the author's personal/university webpage
  b. Try searching "[paper title] filetype:pdf" on Google
  c. Try Semantic Scholar (semanticscholar.org) which often has free PDFs
  d. If still unavailable, note it in 00_PAPER_INDEX.txt as "PAYWALLED — not downloaded"
- Do not download any paper that requires creating an account or logging in
- Work through each step in order, completing each before moving to the next
```

---

## Sources (all research referenced in this document)

- [Customer Restaurant Choice: An Empirical Analysis (Chua et al., 2020) — PMC open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC7503372/)
- [How does the consumer choose a restaurant? Overview of determinants — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0963996924004393)
- [Restaurant Selection Criteria: Roles of restaurant type & sociodemographics — Semantic Scholar](https://www.semanticscholar.org/paper/Restaurant-Selection-Criteria:-Understanding-the-of-Kim-Chung/e54d0bd00cf6e3ad68adbb058f3e23767339841f)
- [Exploring comparative salience of restaurant attributes: conjoint analysis — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0268401216301189)
- [Restaurant Selection Criteria PDF — UMass ScholarWorks](https://scholarworks.umass.edu/bitstreams/b3b5f8a3-2357-4e96-a3e7-3d4e97256b43/download)
- [Food Neophobia Scale — Pliner & Hobden 1992 — PubMed](https://pubmed.ncbi.nlm.nih.gov/1489209/)
- [Food Neophobia Scale — ResearchGate PDF](https://www.researchgate.net/publication/21666930_Development_of_a_scale_to_measure_the_trait_of_food_neophobia_in_humans)
- [User Modeling and User Profiling: A Comprehensive Survey — arXiv](https://arxiv.org/html/2402.09660v2)
- [Explainable Active Learning for Preference Elicitation (2023) — arXiv](https://arxiv.org/abs/2309.00356)
- [Active Learning for Cold Start Recommender Systems (2025) — Nature Scientific Reports](https://www.nature.com/articles/s41598-025-09708-2)
- [Personalized Recommendations via Active Utility-based Pairwise Sampling (2025) — arXiv](https://arxiv.org/html/2508.14911)
- [Cold-start Recommendation by Personalized Embedding Region Elicitation (2024) — arXiv](https://arxiv.org/abs/2406.00973)
- [Asking Clarifying Questions for Preference Elicitation With LLMs (2025) — arXiv](https://arxiv.org/abs/2510.12015)
- [Asking Clarifying Questions — Google Research](https://research.google/pubs/asking-clarifying-questions-for-preference-elicitation-with-large-language-models/)
- [Preference Elicitation Strategy for Conversational Recommender System — ACM WSDM 2019](https://dl.acm.org/doi/10.1145/3289600.3291604)
- [Aspect-based active learning for user preference elicitation — CIRCLE 2020](https://ceur-ws.org/Vol-2621/CIRCLE20_16.pdf)
- [Psycho-Behavioural Segmentation in Food and Nutrition: Systematic Review — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8226652/)
- [User Cold Start Problem in Recommendation Systems: Systematic Review — ResearchGate](https://www.researchgate.net/publication/376140792_User_Cold_Start_Problem_in_Recommendation_Systems_A_Systematic_Review)

---

*Last updated: April 2026*
*Research compiled for Ka'ana Taste ID onboarding design — Phase 1 of recommendation system.*
