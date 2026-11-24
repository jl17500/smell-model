# smell-model

Standalone v1 model for Smell Optimizer: quiz answers about everyday smells → ranked
product recommendations. It is kept separate from the app repo.

See **[PLAN.md](PLAN.md)** for the direction and **[reports/results.md](reports/results.md)**
for the latest numbers. The team write-up is `reports/model_plan.pdf`.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m smell.simulate --n 120 --seed 0   # synthetic form export -> data/synthetic/
.venv/bin/python run.py                                 # fit, compare, write reports/results.md (~3 min)
.venv/bin/python run.py --no-curve                      # skip the sample-size experiment (~40 s)
.venv/bin/python form_experiment.py                     # compare survey designs (~3 min)
```

## Create the survey

Open https://script.google.com → **New project** → paste `form/create_form.gs` → choose
`createSmellForm` → **Run** → allow access. The execution log prints the share link, the edit
link and a linked response sheet. Check the draft notes in `data/real/catalog.csv` first.

## Use real responses

1. Download the responses as CSV (from the form or its sheet) and save them as `data/real/form_responses.csv`.
2. `data/real/catalog.csv` already lists the 16 survey products (columns
   `product_id,name,category,notes,in_survey,source`, notes separated by `;`). Their notes are an
   UNVERIFIED draft: check each one on the brand's page, then add more products.
   Product names must match the rating-grid rows on the form exactly.
3. Run `.venv/bin/python run.py --data data/real --no-curve`.

The loader reads any everyday-smell questions listed in `QUIZ_NODES` (4 labelled choices or a
1–6 linear scale), a grid titled `Rate the products you have smelled` with rows
`[Product name]`, and an optional `Scented products you already love (optional)` question
(one product per line). Titles are set in `smell/data.py`.

## Layout

| file | what it does |
|---|---|
| `smell/taxonomy.py` | the 10 scent nodes; quiz-smell → node and product-note → node mappings |
| `smell/data.py` | Google Form CSV → quiz matrix, product ratings, favorites |
| `smell/models.py` | Popularity (baseline), Note match, Persona clusters (k-means), all with one fit/predict interface |
| `smell/evaluate.py` | split by person, MAE / liked@k, paired bootstrap, cross-validated hyperparameters |
| `smell/simulate.py` | synthetic respondents in the same CSV format; hidden truth in `_truth/` for diagnostics only |
| `run.py` | end-to-end model comparison → `reports/results.md` |
| `form/create_form.gs` | builds the v2 Google Form + response sheet (titles match the loader) |
| `data/real/catalog.csv` | the 16 survey products with draft notes (verify before use) |
| `form_experiment.py` | same people, different survey designs → `reports/form_experiment.md` |

All data in `data/synthetic/` is invented. Its products are fictional.
