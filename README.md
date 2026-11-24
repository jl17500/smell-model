# smell-model

Standalone v1 model for Smell Optimizer: quiz answers about everyday smells → ranked
product recommendations. It is kept separate from the app repo.

See **[PLAN.md](PLAN.md)** for the direction and **[reports/results.md](reports/results.md)**
for the latest numbers.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m smell.simulate --n 120 --seed 0   # synthetic form export -> data/synthetic/
.venv/bin/python run.py                                 # fit, compare, write reports/results.md (~3 min)
.venv/bin/python run.py --no-curve                      # skip the sample-size experiment (~40 s)
```

## Use real responses

1. Download the Google Form responses as CSV and save them as `data/real/form_responses.csv`.
2. Put the product list in `data/real/catalog.csv`, with columns
   `product_id,name,category,notes,in_survey,source` and notes separated by `;`.
   Product names must match the rating-grid rows on the form exactly.
3. Run `.venv/bin/python run.py --data data/real --no-curve`.

The loader expects the six quiz questions as worded on the form, a grid titled
`Rate the products you have smelled` with rows `[Product name]`, and an optional
`Favorite scented product (optional)` question (titles are set in `smell/data.py`).

## Layout

| file | what it does |
|---|---|
| `smell/taxonomy.py` | the 10 scent nodes; quiz-smell → node and product-note → node mappings |
| `smell/data.py` | Google Form CSV → quiz matrix, product ratings, favorites |
| `smell/models.py` | Popularity (baseline), Note match, Persona clusters (k-means), all with one fit/predict interface |
| `smell/evaluate.py` | split by person, MAE / liked@k, paired bootstrap, cross-validated hyperparameters |
| `smell/simulate.py` | synthetic respondents in the same CSV format; hidden truth in `_truth/` for diagnostics only |
| `run.py` | end-to-end comparison and report |

All data in `data/synthetic/` is invented. Its products are fictional.
