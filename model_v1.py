"""Model v1: persona clustering (k-means, k=3) vs a one-group baseline, on the synthetic survey.

    python model_v1.py

Writes charts and printed output to reports/model_v1/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from smell.data import Responses, load_catalog, node_table, parse_form  # noqa: E402
from smell.evaluate import split, user_metrics  # noqa: E402
from smell.models import PersonaCluster, Popularity, TasteProfile  # noqa: E402

DATA = Path("data/synthetic")
OUT = Path("reports/model_v1")
SEED = 0
TEST_FRAC = 0.3
K = 3
K_RANGE = range(1, 9)

# MADE-UP business assumptions, for illustration only.
USERS_PER_MONTH = 5_000
PICKS_PER_USER = 3
PURCHASE_RATE = 0.05  # share of liked picks that turn into a purchase
COMMISSION = 8.0      # dollars per purchase

catalog = load_catalog(DATA / "catalog.csv")
nodes = node_table(catalog)
resp = parse_form(pd.read_csv(DATA / "form_responses.csv"), catalog)
surveyed = sorted(resp.ratings.product_id.unique())
train, test = split(resp, TEST_FRAC, SEED)

# Quiz answers -> taste-profile vectors (centred on the training average of each question).
profile = TasteProfile().fit(train.quiz)
u_train = profile.transform(train, nodes).to_numpy()
u_test = profile.transform(test, nodes).to_numpy()

# Elbow charts: training inertia, and test people's inertia against centres trained at each k.
train_inertia, test_inertia = [], []
for k in K_RANGE:
    km = KMeans(k, n_init=10, random_state=SEED).fit(u_train)
    train_inertia.append(km.inertia_)
    test_inertia.append(((u_test - km.cluster_centers_[km.predict(u_test)]) ** 2).sum())


def elbow_k(inertia):
    """Knee point: k farthest below the straight line from the first to the last point
    (both axes scaled to 0-1)."""
    x, y = np.array(K_RANGE, float), np.asarray(inertia)
    xn, yn = (x - x[0]) / (x[-1] - x[0]), (y - y.min()) / (y.max() - y.min())
    return int(x[np.argmax((1 - xn) - yn)])


def elbow_chart(inertia, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(K_RANGE), inertia, marker="o", color="#2a6fdb")
    ax.axvline(elbow_k(inertia), color="#999", ls="--", lw=1, label=f"elbow k={elbow_k(inertia)}")
    ax.set(xlabel="number of personas (k)", ylabel="inertia (within-group spread)", title=title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


OUT.mkdir(parents=True, exist_ok=True)
elbow_chart(train_inertia, f"Training elbow ({len(train.quiz)} people)", OUT / "elbow_train.png")
elbow_chart(test_inertia, f"Test elbow ({len(test.quiz)} held-out people)", OUT / "elbow_test.png")

# Models: baseline = one group (each product's average rating); v1 = 3 personas.
baseline = Popularity().fit(train, nodes)
v1 = PersonaCluster(k=K, seed=SEED).fit(train, nodes)
results = {name: user_metrics(m.predict(test), test) for name, m in
           [("Baseline (k=1)", baseline), (f"v1 personas (k={K})", v1)]}


def recommend(answers):
    """answers: quiz question -> -2..+2 (unanswered = 0). Returns (persona, top-3 product names)."""
    quiz = pd.DataFrame([{q: answers.get(q, 0) for q in train.quiz.columns}]).rename_axis("respondent")
    person = Responses(quiz, pd.DataFrame(columns=["respondent", "product_id", "rating"]),
                       pd.Series([[]], index=quiz.index))
    persona = int(v1.assign(person)[0])
    top = v1.predict(person).iloc[0][surveyed].nlargest(3).index
    return persona, list(catalog.loc[top, "name"])


lines = [
    "MODEL V1: persona clustering vs one-group baseline (synthetic data)",
    f"Data: {len(resp.quiz)} respondents, {len(resp.ratings)} ratings, {len(surveyed)} survey products",
    f"Split by person (seed {SEED}): {len(train.quiz)} train / {len(test.quiz)} test",
    "",
    "Elbow (inertia by k)",
    "  k   train      test",
    *[f"  {k}  {a:8.1f}  {b:8.1f}" for k, a, b in zip(K_RANGE, train_inertia, test_inertia)],
    f"  elbow: train k={elbow_k(train_inertia)}, test k={elbow_k(test_inertia)}",
    "",
    "Test people: top-3 hit rate (share of top-3 picks rated 4+), mean absolute rating error (1-5 scale)",
]
hit = {}
for name, pu in results.items():
    hit[name] = pu["liked@3"].mean()
    lines.append(f"  {name:20s} top-3 hit {hit[name]:.0%}   MAE {pu.mae.mean():.2f}")

sample = {"Air outside after summer rain": 2, "Clean laundry": 2, "Fresh roses": -2, "Vanilla candle": -2}
persona, picks = recommend(sample)
lines += ["", "Sample user: loves rain + clean laundry, dislikes roses + vanilla",
          f"  persona {persona}; top 3: {'; '.join(picks)}", "",
          "Value per month (MADE-UP assumptions: "
          f"{USERS_PER_MONTH:,} users/month, {PICKS_PER_USER} picks each, {PURCHASE_RATE:.0%} purchase rate "
          f"on liked picks, ${COMMISSION:.0f} commission)"]
for name, h in hit.items():
    value = USERS_PER_MONTH * PICKS_PER_USER * h * PURCHASE_RATE * COMMISSION
    lines.append(f"  {name:20s} ${value:,.0f}/month")

text = "\n".join(lines)
(OUT / "output.txt").write_text(text + "\n")
print(text)
