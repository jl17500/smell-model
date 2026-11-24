"""Load a Google Forms CSV export (real or synthetic) into model-ready tables."""
import re
import warnings
from dataclasses import dataclass

import pandas as pd

from .taxonomy import NODES, QUIZ_ITEMS, node_vector

QUIZ_SCALE = {"extremely dislike": -2, "dislike": -1, "like": 1, "extremely like": 2}
# Proposed second section of the form: a 1-5 grid whose rows are survey products
# (rows optional, so people only rate what they have smelled), plus one free-text question.
RATING_QUESTION = "Rate the products you have smelled"
FAVORITE_QUESTION = "Favorite scented product (optional)"


@dataclass
class Responses:
    quiz: pd.DataFrame     # respondent x quiz question, values -2, -1, 1, 2
    ratings: pd.DataFrame  # long table: respondent, product_id, rating (1-5)
    favorite: pd.Series    # respondent -> product_id; NaN if blank or not in the catalog

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

    quiz = pd.DataFrame({q: form[q].astype(str).str.strip().str.lower().map(QUIZ_SCALE)
                         for q in QUIZ_ITEMS})
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

    text = form.get(FAVORITE_QUESTION, pd.Series("", index=form.index)).fillna("").astype(str).str.strip()
    favorite = text.str.lower().map(by_name)
    unmatched = (text != "") & favorite.isna()
    if unmatched.any():
        warnings.warn(f"{unmatched.sum()} favorite products not found in the catalog")
    return Responses(quiz, ratings.reset_index(drop=True), favorite)
