"""Which survey design predicts best? Same hidden world and the same simulated people
every time; only what the form asks changes. Writes reports/form_experiment.md.

    python form_experiment.py        # ~5 min
"""
from pathlib import Path

import pandas as pd

from run import BASELINE, evaluate, table
from smell.data import Responses, node_table, parse_form
from smell.simulate import simulate
from smell.taxonomy import CURRENT_QUIZ, PROPOSED_QUIZ

N = 120
SEEDS = range(10)
DESIGNS = {
    "A. Current: 6 smells, 4 choices, 1 loved product": dict(questions=CURRENT_QUIZ, scale=4, max_loved=1),
    "B. A + peaches, clean laundry, orange peel": dict(questions=PROPOSED_QUIZ, scale=4, max_loved=1),
    "C. B on a 1-6 scale (no neutral)": dict(questions=PROPOSED_QUIZ, scale=6, max_loved=1),
    "D. C + up to 3 loved products": dict(questions=PROPOSED_QUIZ, scale=6, max_loved=3),
}
COLUMNS = {  # (model, inputs) -> column label
    (BASELINE, "smells only"): "Popularity",
    ("Note match", "smells only"): "Note match: smells only",
    ("Note match", "smells + loved"): "Note match: + loved",
    ("Persona clusters + notes", "smells only"): "Persona+notes: smells only",
    ("Persona clusters + notes", "smells + loved"): "Persona+notes: + loved",
}


def smells_only(resp):
    return Responses(resp.quiz, resp.ratings, resp.favorite.apply(lambda _: []))


def main():
    rows = []
    for seed in SEEDS:
        # Never score a product the person would name as loved, in any design, so every
        # design is graded on exactly the same ratings.
        catalog, form, _ = simulate(N, seed, **DESIGNS["D. C + up to 3 loved products"])
        exclude = parse_form(form, catalog).favorite
        for design, form_args in DESIGNS.items():
            catalog, form, _ = simulate(N, seed, **form_args)
            resp, nodes = parse_form(form, catalog), node_table(catalog)
            for inputs, r in [("smells only", smells_only(resp)), ("smells + loved", resp)]:
                *_, per_user = evaluate(r, nodes, seed, exclude=exclude)
                rows += [{"design": design, "seed": seed, "col": COLUMNS[(m, inputs)],
                          "mae": pu.mae.mean(), "liked@3": pu["liked@3"].mean()}
                         for m, pu in per_user.items() if (m, inputs) in COLUMNS]
    df = pd.DataFrame(rows)

    def pivot(metric):
        return df.pivot_table(index="design", columns="col", values=metric)[list(COLUMNS.values())]

    first = next(iter(DESIGNS))
    mae = df.pivot_table(index=["design", "seed"], columns="col", values="mae")[list(COLUMNS.values())]
    wins = (mae.lt(mae.loc[first].reindex(mae.index, level="seed").to_numpy())
            .groupby("design").sum().astype(str) + f"/{len(SEEDS)}")
    report = "\n\n".join([
        "# Survey design experiment",
        f"{N} simulated respondents per survey, {len(SEEDS)} samples of people, the same hidden world. "
        "Within a sample the people, their tastes and their product ratings are identical across designs; "
        "only the questions asked change. Every design is scored on the same held-out ratings "
        "(products a person names as loved are never scored). 'Smells only' ignores loved products "
        "entirely, which is how the app works for someone who names none.",
        "## Mean MAE (lower is better)", table(pivot("mae").rename_axis("design")),
        "## Mean liked@3 (higher is better)", table(pivot("liked@3").rename_axis("design")),
        f"## Samples where MAE beat design {first[0]} (same people, same model)", table(wins.rename_axis("design")),
    ])
    out = Path("reports/form_experiment.md")
    out.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
