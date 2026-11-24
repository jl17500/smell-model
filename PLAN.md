# Smell Optimizer: model plan

**Goal of the model.** A new user answers a short quiz about everyday smells (plus, optionally,
a product they already love). We rank products they haven't tried so the first few suggestions
are ones they'll like. Everything is built on one shared vocabulary of 10 scent **nodes**
(citrus, fresh, green, floral, fruity, spicy, gourmand, woody, amber, musk). Quiz smells and
product notes both map onto these nodes.

## 1. Fix the data collection first (before sending the form)

The current form only collects **inputs**, the six quiz answers. It never asks anyone to rate a
real product, so there is no **outcome** to check a recommendation against. Without that,
no model can be validated, and the course requires validation on real data against a baseline.

Add to the form before it goes out:

1. **Product rating grid.** "Rate the products you have smelled (leave a row blank if you
   haven't)", scored 1–5, with 12–16 rows of well-known products: half colognes/perfumes, half
   candles, spread across the scent families. Choose products that many people have
   actually smelled (best-sellers), so ratings overlap between people.
2. **Favorite product (optional)**: "Brand + exact product name". We break it into notes and
   use it as an extra clue.
3. Keep the six forced-choice quiz questions as the team decided. Later, consider adding
   near-universal smells for nodes the quiz can't see at all: *fruity* (fresh peaches) and
   *musk/clean* (clean laundry). Citrus is only covered weakly, through ginger. Also check
   cinnamon rolls against vanilla candle in the real answers. Both measure "sweet", and if
   their answers correlate strongly, one of them is a wasted question.
4. One open question: "imagine **wearing** it" fits colognes but not candles. Decide whether the
   quiz is meant to carry over to candles.

**How many responses:** aim for **≥120 people who each rate ≥5 products**. In simulation,
every model beats the baseline reliably only from about 120 respondents. Below 60, the
comparison is close to a coin flip (see `reports/results.md`).

## 2. Catalog: curate, don't scrape the whole web (yet)

A huge scraped database doesn't make the model more accurate until we have ratings to learn
from. It is also slow to build, note data on the web is inconsistent, and many sites forbid
scraping in their terms. Instead:

- Start with **50–100 products** (colognes + candles). For each, record brand, exact name,
  category, published notes, and source URL from the brand page or one consistent public dataset.
- Every note must map to a node (`smell/taxonomy.py`). The loader warns about notes it
  doesn't recognise.
- Grow the catalog only after the model shows it beats the baseline. The node layer means new
  products can be recommended with zero ratings.

## 3. Model ladder (simple first; each step must beat the one before)

| step | model | what it uses | status |
|---|---|---|---|
| baseline | **Popularity**: community average rating per product | ratings only | built |
| dumb model | **Note match**: quiz → node profile → overlap with product notes | hand-made mapping + 1 fitted slope | built |
| v1 | **Persona clusters (+ notes)**: k-means groups people by node profile; recommend what their group rated above average (+ note overlap) | quiz + ratings | built |
| v2 candidates | learn the quiz→product mapping from data (ridge regression); user-kNN on in-app ratings (the "A, B, C liked D" idea = collaborative filtering) | needs more ratings | next |

Your A/B/C/D idea is **collaborative filtering**, and user-based kNN is a fine way to do it.
It needs people who have rated *overlapping products*. The rating grid creates that overlap,
and the app's Rating feature will add more over time. That makes it a v2 model.
Clustering is the coarse version: k-means groups similar people, and the group's
ratings stand in for the neighbours'.

## 4. What the synthetic run says (see `reports/results.md`)

On synthetic data (120 respondents, 20 different train/test splits):

- **Note match beats the popularity baseline on MAE in 20/20 splits**, and Persona clusters +
  notes does in 19/20. The node idea works when the mapping is roughly right.
- **The clusters add no accuracy beyond the notes.** Six 4-point answers carry little
  information, and squeezing people into k groups discards some of it. This held even when we
  made the hand-made mapping 90% wrong. Clusters are still useful for explanations and
  segments (for example, "you're in the fresh & green group").
- k-means doesn't recover the hidden "true" personas well (adjusted Rand index about 0.1–0.2),
  so present personas as useful segments, not as discovered types of people.
- With about 100 people, a single 80/20 split is too noisy to rank models. Always report the
  repeated-splits table.

**Recommendation for the v1 slide:** if the team is reporting a clustering model, use
**Persona clusters + notes** as v1. Show Note match next to it as the ablation, and say plainly
which part delivers the lift. Make the final call on the real data, using the repeated-splits
table.

## 5. Process and stack

- This repo is the **standalone model repo**. Nothing here goes into the app repo (per Grace).
- When it's time to integrate (Dec 1), the app calls one function, answers in and top-5 out,
  with the fitted model saved as a file. Railway hosts the app and database as before.
- Synthetic data is only for checking the pipeline and making design decisions. Once real
  responses arrive, export the form to CSV and run the same command on it (see README).
