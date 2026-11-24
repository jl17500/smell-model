"""Split by person, per-person metrics, paired bootstrap, and hyperparameter search."""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

LIKED = 4      # a rating of 4 or 5 counts as liked
MIN_RATED = 3  # ranking metrics need at least this many rated products per person


def split(resp, test_frac, seed):
    """Hold out whole people: the test set is people the model has never seen."""
    ids = np.random.default_rng(seed).permutation(resp.quiz.index.to_numpy())
    n_test = round(len(ids) * test_frac)
    return resp.subset(ids[n_test:]), resp.subset(ids[:n_test])


def user_metrics(pred, test, exclude=None):
    """Per person: mean absolute error, and whether the products the model ranks
    first (among those the person rated) were ones they liked.

    Products a person named as loved are an input, so they are not also scored as an
    outcome. `exclude` (respondent -> list of product_ids) defaults to test.favorite."""
    r = test.ratings
    fav = (test.favorite if exclude is None else exclude).explode().dropna()
    named = pd.MultiIndex.from_arrays([fav.index, fav.to_numpy()])
    r = r[~pd.MultiIndex.from_arrays([r.respondent, r.product_id]).isin(named)]
    p = pred.to_numpy()[pred.index.get_indexer(r.respondent), pred.columns.get_indexer(r.product_id)]
    r = (r.assign(pred=p, err=np.abs(p - r.rating), liked=(r.rating >= LIKED).astype(float))
         .sort_values(["respondent", "pred"], ascending=[True, False]))
    rank = r.groupby("respondent").cumcount()
    by = r.groupby("respondent")
    out = pd.DataFrame({"mae": by.err.mean(),
                        "liked@1": r[rank == 0].set_index("respondent").liked,
                        "liked@3": r[rank < 3].groupby("respondent").liked.mean()})
    out.loc[by.size() < MIN_RATED, ["liked@1", "liked@3"]] = np.nan
    return out


def paired_bootstrap(a, b, col, n_boot=2000, seed=0):
    """Mean of (a - b) over the same test people, with a 95% bootstrap interval."""
    d = (a[col] - b[col]).dropna().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return d.mean(), lo, hi


def select_params(train, nodes, build, grid, folds=5, seed=0):
    """Pick hyperparameters by cross-validated MAE over people in the training set only."""
    ids = train.quiz.index.to_numpy()
    cv = list(KFold(folds, shuffle=True, random_state=seed).split(ids))
    scores = []
    for params in grid:
        maes = []
        for tr, va in cv:
            val = train.subset(ids[va])
            model = build(**params).fit(train.subset(ids[tr]), nodes)
            maes.append(user_metrics(model.predict(val), val).mae.mean())
        scores.append(np.mean(maes))
    return grid[int(np.argmin(scores))]
