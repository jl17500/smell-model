# Smell Optimizer: model plan

**Goal of the model.** A new user answers a few quick questions about everyday smells (and,
optionally, names products they already love). We rank products they haven't tried so the first
few suggestions are ones they'll like. Everything runs through one shared vocabulary of 10 scent
**nodes** (citrus, fresh, green, floral, fruity, spicy, gourmand, woody, amber, musk). Everyday
smells and product notes both map onto these nodes.

**Inputs vs. answer key.** App users only answer everyday smells; loved products are optional.
Survey respondents answer the same smells *and* rate real products they've smelled. Those ratings
are the answer key that trains and tests the model, never an input for a new user.

The team write-up is `reports/model_plan.pdf` (results there are from the original form design).

## 1. Survey v2 (adopted)

Built by `form/create_form.gs` (run once in Google Apps Script; it creates the form and a linked
response sheet). It has three sections:

1. **Everyday smells (9 questions, required).** The original six plus fresh peaches, clean laundry
   and orange peel, on a 1–6 scale with no neutral. The wording now reads "Imagine a perfume,
   cologne or candle inspired by each smell", so it covers candles too.
2. **Products you have smelled (optional grid, 1–5).** 16 well-known products, 8 fragrances and
   8 candles. This is the answer key.
3. **Products you already love (optional).** Free text, up to 3. Typed names are matched forgivingly
   to the catalog (case, accents, punctuation, small typos, partial names like "Acqua di Gio").

**Target:** at least 120 people who each rate 5 or more products.

**Before sending:** check the draft notes in `data/real/catalog.csv` against each brand's page. They
were written from memory and are marked UNVERIFIED, and the loader warns until they're fixed.

## 2. Catalog: curate, don't scrape the whole web (yet)

Grow `data/real/catalog.csv` to **50–100 products**: brand, exact name, category, published notes,
source URL. Every note must map to a node (`smell/taxonomy.py`); the loader warns about unknown
notes. A web-wide scrape doesn't make the model more accurate until we have ratings. Because
recommendations go through nodes, new products can be recommended with zero ratings.

## 3. Model ladder

| step | model | status |
|---|---|---|
| baseline | **Popularity**: community average rating per product | built |
| dumb model | **Note match**: smells → node profile → overlap with product notes | built |
| v1 | **Persona clusters + notes**: k-means groups people by node profile; recommend what the group rated above average, plus note overlap | built |
| v2 | learned smell → product mapping (ridge regression); user-kNN collaborative filtering on in-app feedback | next |

## 4. What the synthetic runs say (`reports/`)

- On the v2 form, Persona clusters + notes and Note match both beat popularity in 20/20 splits
  (MAE about 0.70 vs 0.79; liked@3 about 0.56 vs 0.47). Clusters + notes is slightly ahead.
- **Loved-product check:** hide a person's loved product and rank all 40 products from smells
  alone. Note-based models put it in the top 10 about 48% of the time (random: 25%). Clusters
  alone score 22%, because they can't rank products nobody rated. That's why v1 is clusters *plus* notes.
- If our smell → node map is badly wrong, the clustering models hold up better than note match.
- In the survey-design test (`reports/form_experiment.md`), the extra smells and the 1–6 scale
  improved accuracy in 8–9 of 10 paired tests. Loved products added about 1 point.

## 5. After real data: levers, biggest first

1. **In-app feedback** (ratings, "Not for me"). It teaches the model directly and enables
   "people like you also liked" (kNN collaborative filtering). The quiz just gets each new user started.
2. **More information per question.** Pairwise questions ("rain or roses?") cancel out people who
   rate everything high. An adaptive quiz chooses the next question based on earlier answers.
3. **Learn the smell → product link from data** (regression v2) instead of trusting our hand-made map.
4. **Cheap MAE fix:** use each person's average quiz answer as a signal of how generously they
   rate. It lowers MAE but barely changes rankings.
5. **Better product data:** verified notes, and base notes (the longest-lasting) weighted more than top notes.
6. **Measure the real smell → product link first.** The simulation can't tell us how strong it is,
   and that number decides which lever is worth the effort.

Why MAE barely moves: people's own ratings are noisy. In the simulation that noise sets a floor
of about 0.6 that no model can beat, so judge the model mainly on liked@3 (are the top picks good?).

## 6. Process and stack

- This is the **standalone model repo**. Nothing here goes into the app repo (per Grace).
- When it's time to integrate (Dec 1), the app calls one function, smell answers in and top 5 out,
  with the fitted model saved as a file. Railway hosts the app and database as before.
