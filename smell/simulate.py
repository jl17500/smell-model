"""Synthetic respondents for the "Smell ratings" Google Form.

Writes files shaped like a Google Forms CSV export, so the same loader runs on real
responses later. Every product and person is invented. The hidden truth (each
person's persona) goes to _truth/ and is used only for diagnostics, never by a model.

How one fake respondent is made:
  1. draw a hidden persona, then a personal taste over scent nodes around it;
  2. answer the everyday-smell questions from that taste. Each smell's *true* node mix
     deliberately differs from our hand-made mapping, and answers are noisy;
  3. rate the survey products they have smelled (popular ones are smelled more often);
  4. sometimes name products they already love.

The form design (which questions, which answer scale, how many loved products) only
changes what is *asked*: the same seed gives the same people with the same tastes and
ratings, so form designs can be compared fairly.

    python -m smell.simulate --n 120 --seed 0 --out data/synthetic
    python -m smell.simulate --questions proposed --scale 6 --max-loved 3 --out data/synthetic_proposed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FAVORITE_QUESTION, RATING_QUESTION, node_table
from .taxonomy import CURRENT_QUIZ, NODES, PROPOSED_QUIZ, QUIZ_NODES, quiz_matrix, to_vector

ANSWERS = ["Extremely dislike", "Dislike", "Like", "Extremely like"]

PERSONAS = {  # hidden taste centres over NODES, with population share
    "fresh & clean": (0.24, dict(citrus=.9, fresh=1.3, green=.5, musk=.6, spicy=-.3, gourmand=-.8, amber=-.7)),
    "sweet tooth": (0.22, dict(gourmand=1.4, fruity=.7, amber=.5, floral=.2, fresh=-.2, green=-.6, woody=-.4)),
    "floral": (0.18, dict(floral=1.4, fruity=.8, musk=.6, spicy=-.5, woody=-.4, amber=-.3)),
    "warm & spicy": (0.20, dict(woody=1.1, spicy=1.1, amber=1.0, gourmand=.4, fresh=-.3, floral=-.7, fruity=-.6)),
    "outdoorsy": (0.16, dict(green=1.2, woody=.9, fresh=.7, citrus=.4, floral=-.4, gourmand=-.7)),
}

# (product family, category, products in the catalog, note pool; the first two notes are always used).
# The first product of each family is put on the survey, so 16 survey products span the node space.
FAMILIES = [
    ("Citrus Aromatic", "cologne", 3, ["bergamot", "lemon", "grapefruit", "lavender", "rosemary", "vetiver", "musk"]),
    ("Fresh Aquatic", "cologne", 3, ["sea salt", "marine", "bergamot", "ozone", "mint", "cedar", "ambroxan"]),
    ("Woody Spice", "cologne", 3, ["cardamom", "cedar", "pink pepper", "black pepper", "vetiver", "sandalwood", "ginger"]),
    ("Amber Night", "cologne", 3, ["amber", "incense", "tonka", "vanilla", "tobacco", "labdanum", "cinnamon"]),
    ("Rose Garden", "cologne", 3, ["rose", "peony", "pink pepper", "musk", "raspberry", "patchouli", "peach"]),
    ("Sweet Vanilla", "cologne", 3, ["vanilla", "caramel", "praline", "coffee", "tonka", "almond", "jasmine"]),
    ("Green Fougere", "cologne", 3, ["lavender", "oakmoss", "basil", "sage", "fig leaf", "vetiver", "tonka"]),
    ("Fruity Floral", "cologne", 3, ["peach", "pear", "blackcurrant", "jasmine", "magnolia", "musk", "sandalwood"]),
    ("Christmas Forest", "candle", 2, ["fir", "pine", "cedar", "cinnamon", "clove", "orange"]),
    ("Bakery", "candle", 2, ["cinnamon", "brown sugar", "vanilla", "buttercream", "nutmeg", "caramel"]),
    ("Clean Linen", "candle", 2, ["clean cotton", "white musk", "lemon", "lavender", "ozone"]),
    ("Garden Bloom", "candle", 2, ["rose", "peony", "jasmine", "grass", "pear"]),
    ("Rain & Moss", "candle", 2, ["rain", "oakmoss", "vetiver", "ozone", "sage", "cedar"]),
    ("Vanilla Amber", "candle", 2, ["vanilla", "amber", "sandalwood", "tonka", "musk"]),
    ("Citrus Grove", "candle", 2, ["orange", "grapefruit", "lime", "mint", "basil"]),
    ("Spiced Chai", "candle", 2, ["tea", "ginger", "cardamom", "honey", "lemon"]),
]

# How messy the hidden world is. Each value is an assumption to check against real responses.
TASTE_SD = 0.5             # spread of people within a persona
QUIZ_MAPPING_ERROR = 0.25  # share of each everyday smell's true node mix that our mapping misses
PRODUCT_HIDDEN = 0.15      # share of a product's character its listed notes don't capture
QUIZ_NOISE = 0.7
RATING_NOISE = 0.6
QUIZ_BIAS = {  # "imagine wearing it": food smells appeal less as a perfume than as food
    "Fresh roses": 0.0, "Christmas tree (Evergreen)": 0.0, "Air outside after summer rain": 0.4,
    "Freshly baked cinnamon rolls": -0.3, "Vanilla candle": 0.3, "Freshly sliced ginger": -0.3,
    "Fresh peaches": 0.0, "Clean laundry": 0.4, "Orange peel": 0.1,
}
FAMILIARITY = {  # chance someone really knows the smell; otherwise the forced choice is a guess
    "Fresh roses": 0.9, "Christmas tree (Evergreen)": 0.9, "Air outside after summer rain": 0.85,
    "Freshly baked cinnamon rolls": 0.95, "Vanilla candle": 0.95, "Freshly sliced ginger": 0.7,
    "Fresh peaches": 0.9, "Clean laundry": 0.95, "Orange peel": 0.9,
}
STRAIGHTLINE_RATE = 0.05   # people who click "Like" for everything
LOVED_RATE = 0.5           # people who name at least one product they love (up to 3)


def make_catalog(rng):
    rows = []
    for family, category, count, pool in FAMILIES:
        for i in range(count):
            extra = rng.choice(pool[2:], size=rng.integers(2, 4), replace=False)
            rows.append({"name": f"{family} No. {i + 1}", "category": category,
                         "notes": pool[:2] + list(extra), "in_survey": i == 0, "source": "synthetic"})
    catalog = pd.DataFrame(rows)
    catalog.index = pd.Index([f"P{i + 1:02d}" for i in range(len(catalog))], name="product_id")
    return catalog


def simulate(n, seed=0, world_seed=0, questions=CURRENT_QUIZ, scale=4, max_loved=1):
    """Returns (catalog, form export, hidden truth) for n respondents.

    world_seed fixes the catalog and hidden truths; seed draws the people. The form
    design: `questions` asked, answer `scale` (4 labelled choices or a 1-6 scale; neither
    has a neutral middle), and up to `max_loved` products people may name."""
    world = np.random.default_rng(world_seed)
    catalog = make_catalog(world)
    nodes = node_table(catalog).to_numpy()
    all_q = list(QUIZ_NODES)
    n_nodes, n_q, n_p = len(NODES), len(all_q), len(catalog)

    # Hidden world: true smell and product profiles, product quality and popularity.
    quiz_true = (1 - QUIZ_MAPPING_ERROR) * quiz_matrix(all_q) + QUIZ_MAPPING_ERROR * world.dirichlet(np.full(n_nodes, .5), n_q)
    prod_true = (1 - PRODUCT_HIDDEN) * nodes + PRODUCT_HIDDEN * world.dirichlet(np.full(n_nodes, .5), n_p)
    quality = world.normal(0, 0.35, n_p)
    popularity = world.lognormal(0, 0.7, n_p)

    # People (a fresh sample for each seed, from the same world). Every random draw below
    # is made for all questions and up to 3 loved products, whatever the form design.
    rng = np.random.default_rng(seed)
    names = list(PERSONAS)
    persona = rng.choice(len(names), n, p=[PERSONAS[p][0] for p in names])
    taste = np.array([to_vector(PERSONAS[p][1]) for p in names])[persona] + rng.normal(0, TASTE_SD, (n, n_nodes))
    leniency = rng.normal(0, 0.4, n)                  # some people rate everything higher
    style = 0.5 * leniency + rng.normal(0, 0.2, n)    # ...and answer the quiz more positively

    # Quiz answers: forced choice with no neutral middle.
    bias = np.array([QUIZ_BIAS[q] for q in all_q])
    knows = rng.random((n, n_q)) < np.array([FAMILIARITY[q] for q in all_q])
    latent = 1.2 * taste @ quiz_true.T + rng.normal(0, QUIZ_NOISE, (n, n_q))
    guess = rng.normal(0, 1, (n, n_q))
    score = np.where(knows, latent, guess) + bias + style[:, None]
    straightline = rng.random(n) < STRAIGHTLINE_RATE

    # Product ratings for the survey products each person has smelled.
    utility = 1.2 * taste @ prod_true.T + quality
    survey = np.flatnonzero(catalog.in_survey)
    weights = popularity[survey] / popularity[survey].sum()
    grid = np.full((n, len(survey)), np.nan)
    for i in range(n):
        smelled = rng.choice(len(survey), 3 + rng.binomial(len(survey) - 3, 0.45), replace=False, p=weights)
        z = 3 + utility[i, survey[smelled]] + leniency[i] + rng.normal(0, RATING_NOISE, len(smelled))
        grid[i, smelled] = np.clip(np.rint(z), 1, 5)

    # Products people already love: well-known ones that suit their taste.
    n_loved = np.where(rng.random(n) < LOVED_RATE, 1 + rng.binomial(2, 0.5, n), 0)
    loved = []
    for i in range(n):
        p = popularity * np.exp(2 * utility[i])
        picks = rng.choice(n_p, n_loved[i], replace=False, p=p / p.sum())
        loved.append("; ".join(catalog["name"].iloc[picks[:max_loved]]))

    start = pd.Timestamp("2026-11-20 09:00")
    times = start + pd.to_timedelta(np.sort(rng.uniform(0, 3 * 24 * 60, n)), unit="min")
    form = pd.DataFrame({"Timestamp": times.strftime("%m/%d/%Y %H:%M:%S")})
    for q in questions:
        s = score[:, all_q.index(q)]
        if scale == 4:
            level = np.digitize(s, [-0.9, 0, 0.9])
            level[straightline] = ANSWERS.index("Like")
            form[q] = np.array(ANSWERS)[level]
        else:
            level = 1 + np.digitize(s, [-1.4, -0.6, 0, 0.6, 1.4])
            level[straightline] = 5
            form[q] = level
    for k, pid in enumerate(catalog.index[survey]):
        form[f"{RATING_QUESTION} [{catalog['name'][pid]}]"] = pd.Series(grid[:, k]).astype("Int64")
    form[FAVORITE_QUESTION] = loved

    truth = pd.DataFrame({"persona": np.array(names)[persona]}, index=pd.RangeIndex(n, name="respondent"))
    return catalog, form, truth


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions", choices=["current", "proposed"], default="current")
    ap.add_argument("--scale", type=int, choices=[4, 6], default=4)
    ap.add_argument("--max-loved", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = ap.parse_args()
    questions = CURRENT_QUIZ if args.questions == "current" else PROPOSED_QUIZ
    catalog, form, truth = simulate(args.n, args.seed, questions=questions, scale=args.scale,
                                    max_loved=args.max_loved)
    (args.out / "_truth").mkdir(parents=True, exist_ok=True)
    catalog.assign(notes=catalog["notes"].str.join(";")).to_csv(args.out / "catalog.csv")
    form.to_csv(args.out / "form_responses.csv", index=False)
    truth.to_csv(args.out / "_truth" / "personas.csv")
    print(f"wrote {len(form)} responses and {len(catalog)} products to {args.out}")


if __name__ == "__main__":
    main()
