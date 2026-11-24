"""Load a Google Forms CSV export (real or synthetic) into model-ready tables."""
import re
import warnings
from dataclasses import dataclass

import pandas as pd

from .taxonomy import NODES, QUIZ_NODES, node_vector

# Quiz answers: either the 4 labelled choices, or a 1-6 linear scale (no neutral middle).
# Both land on roughly -2..+2.
QUIZ_SCALE = {"extremely dislike": -2, "dislike": -1, "like": 1, "extremely like": 2}
# Answer key, used only to train and test, never as an input for a new user: a 1-5 grid
# of survey products (rows optional, so people only rate what they have smelled).
RATING_QUESTION = "Rate the products you have smelled"
# Optional input: products the person already loves, one per line or separated by ";".
FAVORITE_QUESTION = "Scented products you already love (optional)"


@dataclass
class Responses:
    quiz: pd.DataFrame     # respondent x quiz question, values roughly -2..+2
    ratings: pd.DataFrame  # long table: respondent, product_id, rating (1-5)
    favorite: pd.Series    # respondent -> list of product_ids (empty if none named)

    def subset(self, ids):
        return Responses(self.quiz.loc[ids],
                         self.ratings[self.ratings.respondent.isin(ids)],
                         self.favorite.loc[ids])


def load_catalog(path):
    catalog = pd.read_csv(path, index_col="product_id")
    catalog["notes"] = catalog["notes"].str.split(";")
    return catalog


def node_table(catalog):
    """Product x node weights."""
    unknown = set()
    rows = [node_vector(notes, unknown) for notes in catalog["notes"]]
    if unknown:
        warnings.warn(f"notes with no node mapping (add them to NOTE_NODES): {sorted(unknown)}")
    return pd.DataFrame(rows, index=catalog.index, columns=NODES)


def parse_form(form, catalog):
    # Form titles carry stray spaces ("Fresh roses "), so normalise whitespace.
    form = form.rename(columns=lambda c: " ".join(str(c).split())).reset_index(drop=True)
    form.index.name = "respondent"

    items = [q for q in QUIZ_NODES if q in form.columns]
    if not items:
        raise ValueError("no known quiz questions in the form (see QUIZ_NODES)")
    quiz = pd.DataFrame({q: _quiz_values(form[q]) for q in items})
    if quiz.isna().any().any():
        raise ValueError("quiz has blank or unrecognised answers")

    by_name = {name.lower(): pid for pid, name in catalog["name"].items()}
    grid = {c: re.search(r"\[(.*)\]$", c).group(1) for c in form if c.startswith(RATING_QUESTION)}
    missing = [name for name in grid.values() if name.lower() not in by_name]
    if missing:
        raise ValueError(f"survey products missing from the catalog: {missing}")
    ratings = (form[list(grid)].rename(columns={c: by_name[n.lower()] for c, n in grid.items()})
               .reset_index()
               .melt(id_vars="respondent", var_name="product_id", value_name="rating")
               .dropna(subset=["rating"]))
    ratings["rating"] = ratings["rating"].astype(int)
    if not ratings.rating.between(1, 5).all():
        raise ValueError("product ratings must be 1-5")

    text = form.get(FAVORITE_QUESTION, pd.Series("", index=form.index)).fillna("").astype(str)
    named = text.str.split(r"[;\n]").apply(lambda xs: [x.strip().lower() for x in xs if x.strip()])
    favorite = named.apply(lambda xs: [by_name[x] for x in xs if x in by_name])
    unmatched = (named.str.len() - favorite.str.len()).sum()
    if unmatched:
        warnings.warn(f"{unmatched} named products not found in the catalog")
    return Responses(quiz, ratings.reset_index(drop=True), favorite)


def _quiz_values(col):
    if pd.api.types.is_numeric_dtype(col):  # 1-6 linear scale
        return (col - 3.5) * 0.8
    return col.astype(str).str.strip().str.lower().map(QUIZ_SCALE)
