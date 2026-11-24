# Smell Optimizer: model plan

**Goal of the model.** A new user answers a few quick questions about everyday smells (and,
optionally, names products they already love). We rank products they haven't tried so the first
few suggestions are ones they'll like. Everything runs through one shared vocabulary of 10 scent
**nodes** (citrus, fresh, green, floral, fruity, spicy, gourmand, woody, amber, musk). Everyday
smells and product notes both map onto these nodes.

**Inputs vs. answer key.** App users only answer everyday smells. Loved products are optional.
Survey respondents answer the same smells *and* rate real products they've smelled. Those ratings
are the answer key: they train and test the model, and they are never an input for a new user.

The full write-up for the team is in `reports/model_plan.pdf`.

## 1. Survey changes before it goes out

1. **Product rating grid (must).** "Rate the products you have smelled (leave a row blank if you
   haven't)", scored 1–5, with 12–16 rows of well-known products: half colognes/perfumes, half
   candles, spread across the scent families.
2. **Three more everyday smells (recommended):** fresh peaches (fruity), clean laundry
   (musk/clean), and orange peel (citrus). The current six can't see fruity or musk at all.
3. **A 1–6 scale with no neutral (recommended).** It captures more per answer and keeps the team's
   rule against neutral answers.
4. **Loved products (optional):** "Scented products you already love": brand + exact name, one per line.
5. Open question: "imagine **wearing** it" fits colognes but not candles.

In simulation, items 2 and 3 improved accuracy in 8–9 of 10 paired tests. Loved products changed
results by only about 1 point, so the model works from smells alone (`reports/form_experiment.md`).

**How many responses:** at least **120 people who each rate at least 5 products**. The clustering
model is only reliable from about 120 respondents (`reports/results.md`).

## 2. Catalog: curate, don't scrape the whole web (yet)

Start with **50–100 products** (colognes + candles): brand, exact name, category, published notes,
and source URL. Every note must map to a node (`smell/taxonomy.py`); the loader warns about notes
it doesn't recognise. A web-wide scrape doesn't make the model more accurate until we have ratings,
it's slow, and many sites forbid scraping. Because recommendations go through nodes, new products
can be recommended with zero ratings.

## 3. Model ladder

| step | model | status |
|---|---|---|
| baseline | **Popularity**: community average rating per product | built |
| dumb model | **Note match**: smells → node profile → overlap with product notes | built |
| v1 | **Persona clusters + notes**: k-means groups people by node profile; recommend what the group rated above average, plus note overlap | built |
| v2 | learned smell → product mapping (ridge regression); user-kNN collaborative filtering on in-app ratings | next |

## 4. What the synthetic runs say

- Note match beats popularity in 20/20 splits on MAE and liked@3. Persona clusters + notes does too,
  essentially tying note match.
- If our smell → node mapping is badly wrong (90%), note match falls to the baseline and the
  clustering models edge ahead, because they learn from people's ratings. That makes
  **Persona clusters + notes the safest v1**. It also gives personas to explain recommendations.
- The ceiling comes from how much the survey tells us, not from the algorithm. Personas are fuzzy
  (adjusted Rand index 0.1–0.2), so present them as segments, not as discovered "scent types."
- With about 100 people, a single 80/20 split is too noisy. Always report repeated splits.

## 5. Process and stack

- This is the **standalone model repo**. Nothing here goes into the app repo (per Grace).
- When it's time to integrate (Dec 1), the app calls one function, smell answers in and top 5 out,
  with the fitted model saved as a file. Railway hosts the app and database as before.
- All numbers so far come from synthetic data. Once real responses arrive, run the same commands on
  the real CSV (see README).
