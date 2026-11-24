"""Load a Google Forms CSV export (real or synthetic) into model-ready tables."""
import difflib
import re
import unicodedata
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
    unverified = catalog["source"].astype(str).str.contains("UNVERIFIED")
    if unverified.any():
        warnings.warn(f"{unverified.sum()} products have unverified notes; check them before trusting results")
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
    named = text.str.split(r"[;\n]").apply(lambda xs: [x for x in xs if x.strip()])
    favorite = named.apply(lambda xs: [pid for pid in (match_product(x, catalog) for x in xs) if pid])
    unmatched = (named.str.len() - favorite.str.len()).sum()
    if unmatched:
        warnings.warn(f"{unmatched} named products not found in the catalog")
    return Responses(quiz, ratings.reset_index(drop=True), favorite)


def _norm(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())


def match_product(text, catalog, cutoff=0.85):
    """Catalog product_id for a typed product name, forgiving case, accents, punctuation
    and small typos; None if nothing is close enough."""
    names = {_norm(n): pid for pid, n in catalog["name"].items()}
    typed = _norm(text)
    hit = difflib.get_close_matches(typed, list(names), n=1, cutoff=cutoff)
    if hit:
        return names[hit[0]]
    # "Acqua di Gio", "Sauvage by Dior": every typed word appears in exactly one product name.
    words = set(typed.split()) - {"by", "the"}
    if len(words) >= 2:
        found = [pid for name, pid in names.items() if words <= set(name.split())]
        if len(found) == 1:
            return found[0]
    return None


def _quiz_values(col):
    if pd.api.types.is_numeric_dtype(col):  # 1-6 linear scale
        return (col - 3.5) * 0.8
    return col.astype(str).str.strip().str.lower().map(QUIZ_SCALE)
