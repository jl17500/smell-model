"""Recommenders. Each has fit(train, nodes) -> self and predict(responses) -> DataFrame of
predicted 1-5 ratings (respondent x product); a person's ranking is that row sorted.
`nodes` is the product x scent-node table from data.node_table()."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .taxonomy import QUIZ_MATRIX

PRIOR = 5              # pseudo-ratings pulling thinly rated products toward the overall mean
FAVORITE_WEIGHT = 2.0  # a named favorite counts about as much as one strong quiz answer


def lookup(table, ratings):
    """Values of a respondent x product table at each row of a ratings table."""
    return table.to_numpy()[table.index.get_indexer(ratings.respondent),
                            table.columns.get_indexer(ratings.product_id)]


class Popularity:
    """Baseline: everyone gets each product's community average rating."""

    def fit(self, train, nodes):
        r = train.ratings
        overall = r.rating.mean()
        g = r.groupby("product_id").rating.agg(["sum", "count"]).reindex(nodes.index, fill_value=0)
        self.mean = (g["sum"] + PRIOR * overall) / (g["count"] + PRIOR)
        return self

    def predict(self, resp):
        return pd.DataFrame(np.tile(self.mean.to_numpy(), (len(resp.quiz), 1)),
                            index=resp.quiz.index, columns=self.mean.index)


class TasteProfile:
    """Quiz answers (+ named favorite) -> preference weights over scent nodes.

    Each answer is centred on that question's training average, so a smell nearly
    everyone likes says less about a person than one that splits people."""

    def fit(self, quiz):
        self.item_mean = quiz.mean()
        return self

    def transform(self, resp, nodes):
        u = (resp.quiz - self.item_mean).to_numpy() @ QUIZ_MATRIX
        fav = resp.favorite.dropna()
        u[resp.quiz.index.get_indexer(fav.index)] += FAVORITE_WEIGHT * nodes.loc[fav].to_numpy()
        return pd.DataFrame(u, index=resp.quiz.index, columns=nodes.columns)


class NoteMatch:
    """Simple content model: products whose notes overlap the person's liked nodes score
    higher. One slope, fit on training ratings, converts overlap into rating points."""

    def fit(self, train, nodes):
        self.nodes = nodes
        self.pop = Popularity().fit(train, nodes)
        self.profile = TasteProfile().fit(train.quiz)
        r = train.ratings
        s = lookup(self.match(train), r)
        resid = r.rating.to_numpy() - self.pop.mean[r.product_id].to_numpy()
        self.slope = s @ resid / (s @ s)
        return self

    def match(self, resp):
        return self.profile.transform(resp, self.nodes) @ self.nodes.T

    def predict(self, resp):
        return self.pop.predict(resp) + self.slope * self.match(resp)


class PersonaCluster:
    """k-means groups people into taste personas by their node profile. A persona's
    prediction for a product = community average + the persona's average deviation
    from it, shrunk toward zero when few members rated that product.

    use_notes=True adds the NoteMatch term, so products nobody in the survey rated
    (most of a real catalog) still get personalised scores."""

    def __init__(self, k=4, shrink=10, use_notes=False, seed=0):
        self.k, self.shrink, self.use_notes, self.seed = k, shrink, use_notes, seed

    def fit(self, train, nodes):
        self.nodes = nodes
        self.pop = Popularity().fit(train, nodes)
        self.profile = TasteProfile().fit(train.quiz)
        u = self.profile.transform(train, nodes)
        self.kmeans = KMeans(self.k, n_init=10, random_state=self.seed).fit(u.to_numpy())
        persona = pd.Series(self.kmeans.labels_, index=u.index)

        r = train.ratings
        r = r.assign(persona=persona[r.respondent].to_numpy(),
                     resid=r.rating.to_numpy() - self.pop.mean[r.product_id].to_numpy())
        g = (r.groupby(["persona", "product_id"]).resid.agg(["sum", "count"])
             .reindex(pd.MultiIndex.from_product([range(self.k), nodes.index]), fill_value=0))
        self.deviation = (g["sum"] / (g["count"] + self.shrink)).unstack().reindex(columns=nodes.index)

        self.slope = 0.0
        if self.use_notes:
            # Leave each rating out of its own persona's deviation, so the slope only
            # learns what the notes add beyond the persona.
            key = pd.MultiIndex.from_arrays([r.persona, r.product_id])
            total, count = g["sum"].reindex(key).to_numpy(), g["count"].reindex(key).to_numpy()
            target = r.resid.to_numpy() - (total - r.resid.to_numpy()) / (count - 1 + self.shrink)
            s = lookup(u @ nodes.T, r)
            self.slope = s @ target / (s @ s)
        return self

    def assign(self, resp):
        return self.kmeans.predict(self.profile.transform(resp, self.nodes).to_numpy())

    def predict(self, resp):
        u = self.profile.transform(resp, self.nodes)
        pred = self.pop.predict(resp) + self.deviation.to_numpy()[self.kmeans.predict(u.to_numpy())]
        return pred + self.slope * (u @ self.nodes.T)
