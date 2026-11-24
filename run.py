"""Fit and compare the recommenders on a form export, then write reports/results.md.

    python run.py                      # synthetic data in data/synthetic
    python run.py --data data/real     # same pipeline on the real form export
    python run.py --no-curve           # skip the how-many-responses experiment
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from smell.data import load_catalog, node_table, parse_form
from smell.evaluate import paired_bootstrap, select_params, split, user_metrics
from smell.models import NoteMatch, PersonaCluster, Popularity
from smell.simulate import simulate

TEST_FRAC = 0.2
GRID = [dict(k=k, shrink=s) for k in range(2, 8) for s in (5, 20)]
BASELINE = "Popularity (baseline)"
CURVE_SIZES = [30, 60, 120, 250, 500]
CURVE_SEEDS = range(10)


def fit_all(train, nodes, seed):
    def persona_model(use_notes):
        build = lambda **p: PersonaCluster(**p, use_notes=use_notes, seed=seed)  # noqa: E731
        return build(**select_params(train, nodes, build, GRID, seed=seed))

    models = {BASELINE: Popularity(), "Note match": NoteMatch(),
              "Persona clusters": persona_model(False), "Persona clusters + notes": persona_model(True)}
    return {name: m.fit(train, nodes) for name, m in models.items()}


def evaluate(resp, nodes, seed, exclude=None):
    """exclude: respondent -> products not to score (defaults to each person's named loved products)."""
    train, test = split(resp, TEST_FRAC, seed)
    models = fit_all(train, nodes, seed)
    ex = None if exclude is None else exclude.loc[test.quiz.index]
    return train, test, models, {name: user_metrics(m.predict(test), test, ex) for name, m in models.items()}


def table(df, fmt="{:.3f}"):
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    rule = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, float) else str(v) for v in row]) + " |"
            for i, row in zip(df.index, df.itertuples(index=False))]
    return "\n".join([head, rule, *body])


def results_section(per_user):
    base = per_user[BASELINE]
    rows = {}
    for name, pu in per_user.items():
        row = {"MAE (pts)": pu.mae.mean(), "liked@1": pu["liked@1"].mean(), "liked@3": pu["liked@3"].mean()}
        if name != BASELINE:
            for col, label in [("mae", "MAE vs baseline"), ("liked@3", "liked@3 vs baseline")]:
                d, lo, hi = paired_bootstrap(pu, base, col)
                row[label] = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        else:
            row["MAE vs baseline"] = row["liked@3 vs baseline"] = "-"
        rows[name] = row
    return table(pd.DataFrame(rows).T.rename_axis("model"))


def persona_section(model, train, nodes, catalog, truth):
    u = model.profile.transform(train, nodes)
    labels = model.kmeans.labels_
    lines = [f"k = {model.k} personas (shrink = {model.shrink}), chosen by 5-fold cross-validation on training people.",
             f"Silhouette = {silhouette_score(u, labels):.2f} (0 = overlapping groups, 1 = cleanly separated)."]
    if truth is not None:
        ari = adjusted_rand_score(truth.loc[train.quiz.index, "persona"], labels)
        lines.append(f"Agreement with the hidden synthetic personas (adjusted Rand index) = {ari:.2f} (1 = perfect).")
    surveyed = train.ratings.product_id.unique()
    centers = pd.DataFrame(model.kmeans.cluster_centers_, columns=nodes.columns)
    rows = []
    for c in range(model.k):
        likes = centers.loc[c][centers.loc[c] > 0.15].nlargest(2).index
        dislikes = centers.loc[c][centers.loc[c] < -0.15].nsmallest(2).index
        top = (model.pop.mean + model.deviation.loc[c])[surveyed].nlargest(3).index
        rows.append({"people": int((labels == c).sum()), "leans toward": ", ".join(likes) or "-",
                     "leans away from": ", ".join(dislikes) or "-",
                     "top survey products": "; ".join(catalog.loc[top, "name"])})
    return "\n\n".join(lines + [table(pd.DataFrame(rows).rename_axis("persona"))])


def learning_curve():
    rows = []
    for n in CURVE_SIZES:
        for seed in CURVE_SEEDS:
            catalog, form, _ = simulate(n, seed)
            *_, per_user = evaluate(parse_form(form, catalog), node_table(catalog), seed)
            rows += [{"n": n, "seed": seed, "model": name, "mae": pu.mae.mean(), "liked@3": pu["liked@3"].mean()}
                     for name, pu in per_user.items()]
    df = pd.DataFrame(rows)
    mae = df.pivot_table(index="n", columns="model", values="mae")
    liked = df.pivot_table(index="n", columns="model", values="liked@3")
    base = df[df.model == BASELINE].set_index(["n", "seed"]).mae
    wins = (df[df.model != BASELINE].set_index(["n", "seed"])
            .assign(win=lambda d: d.mae < base.reindex(d.index).to_numpy())
            .groupby(["n", "model"]).win.sum().unstack().astype(int).astype(str) + f"/{len(CURVE_SEEDS)}")
    return "\n\n".join([
        f"Same hidden world, a fresh sample of people for each size and seed ({len(CURVE_SEEDS)} seeds), {TEST_FRAC:.0%} of people held out each time.",
        "**Mean MAE (lower is better)**", table(mae.rename_axis("respondents")),
        "**Mean liked@3 (higher is better)**", table(liked.rename_axis("respondents")),
        "**Seeds where the model's MAE beat the baseline's**", table(wins.rename_axis("respondents"))])


def repeated_splits(resp, nodes, seeds):
    """The same data split into train/test many ways: is the ranking of models stable?"""
    rows = []
    for seed in seeds:
        *_, per_user = evaluate(resp, nodes, seed)
        rows += [{"split": seed, "model": name, "mae": pu.mae.mean(), "liked@3": pu["liked@3"].mean()}
                 for name, pu in per_user.items()]
    df = pd.DataFrame(rows)
    mae = df.pivot(index="split", columns="model", values="mae")
    liked = df.pivot(index="split", columns="model", values="liked@3")
    out = pd.DataFrame({
        "mean MAE": mae.mean(),
        "MAE beat baseline": mae.lt(mae[BASELINE], axis=0).sum().astype(str) + f"/{len(seeds)}",
        "mean liked@3": liked.mean(),
        "liked@3 beat baseline": liked.gt(liked[BASELINE], axis=0).sum().astype(str) + f"/{len(seeds)}",
    })
    out.loc[BASELINE, ["MAE beat baseline", "liked@3 beat baseline"]] = "-"
    return f"{len(seeds)} different random person splits of this same dataset.\n\n" + table(out.rename_axis("model"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("reports/results.md"))
    ap.add_argument("--no-curve", action="store_true")
    args = ap.parse_args()

    catalog = load_catalog(args.data / "catalog.csv")
    nodes = node_table(catalog)
    resp = parse_form(pd.read_csv(args.data / "form_responses.csv"), catalog)
    truth_path = args.data / "_truth" / "personas.csv"
    truth = pd.read_csv(truth_path, index_col="respondent") if truth_path.exists() else None

    train, test, models, per_user = evaluate(resp, nodes, args.seed)
    n_rated = resp.ratings.groupby("respondent").size()
    report = [
        f"# Model comparison: `{args.data}`",
        "## Data",
        f"- {len(resp.quiz)} respondents; {len(resp.ratings)} product ratings "
        f"(median {n_rated.median():.0f} per person) on {resp.ratings.product_id.nunique()} survey products; "
        f"catalog of {len(catalog)} products.",
        f"- {len(resp.quiz.columns)} everyday-smell questions; {resp.favorite.str.len().gt(0).mean():.0%} named "
        "at least one product they already love (optional input).",
        f"- Split by person: {len(train.quiz)} train / {len(test.quiz)} test (seed {args.seed}). "
        "Hyperparameters are chosen by cross-validation inside train; test is scored once.",
        "## Results on held-out people",
        "MAE = average miss in rating points (1-5 scale). liked@1 / liked@3 = share of the model's top 1 / top 3 "
        "picks, among products the person rated, that they rated 4 or 5. Brackets are 95% paired bootstrap "
        "intervals; an interval that crosses 0 means the difference is not reliable at this sample size.",
        results_section(per_user),
        "## Same data, repeated splits",
        repeated_splits(resp, nodes, range(20)),
        "## Personas found by k-means",
        persona_section(models["Persona clusters"], train, nodes, catalog, truth),
    ]
    if not args.no_curve:
        report += ["## How many responses do we need?", learning_curve()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n".join(report) + "\n")
    print("\n\n".join(report))


if __name__ == "__main__":
    main()
