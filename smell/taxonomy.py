"""Scent nodes and the two hand-made mappings the models rely on.

Everyday smells in the quiz and the notes listed on products are both expressed
as weights over the same small set of scent nodes. Both mappings are hypotheses:
the product ratings we collect are what tell us where they are wrong.
"""
import numpy as np

NODES = ["citrus", "fresh", "green", "floral", "fruity",
         "spicy", "gourmand", "woody", "amber", "musk"]

# Quiz question (as worded in the Google Form) -> node weights (each row sums to 1).
QUIZ_NODES = {
    # Current form. Nothing here covers "fruity" or "musk", and citrus only weakly.
    "Fresh roses": {"floral": 0.9, "green": 0.1},
    "Christmas tree (Evergreen)": {"woody": 0.5, "green": 0.3, "amber": 0.2},
    "Air outside after summer rain": {"fresh": 0.7, "green": 0.3},
    "Freshly baked cinnamon rolls": {"gourmand": 0.6, "spicy": 0.4},
    "Vanilla candle": {"gourmand": 0.8, "amber": 0.2},
    "Freshly sliced ginger": {"spicy": 0.6, "citrus": 0.25, "fresh": 0.15},
    # Proposed additions: near-universal smells for the nodes above that are missing.
    "Fresh peaches": {"fruity": 0.85, "floral": 0.15},
    "Clean laundry": {"musk": 0.6, "fresh": 0.4},
    "Orange peel": {"citrus": 0.9, "fresh": 0.1},
}
CURRENT_QUIZ = list(QUIZ_NODES)[:6]
PROPOSED_QUIZ = list(QUIZ_NODES)

# Product note -> node weights. Extend this as the catalog grows; node_vector()
# reports notes it does not know instead of silently dropping them.
NOTE_NODES = {
    "bergamot": {"citrus": 1}, "lemon": {"citrus": 1}, "orange": {"citrus": 1},
    "grapefruit": {"citrus": 1}, "mandarin": {"citrus": 1}, "lime": {"citrus": 1},
    "sea salt": {"fresh": 1}, "marine": {"fresh": 1}, "ozone": {"fresh": 1},
    "rain": {"fresh": 0.7, "green": 0.3}, "mint": {"fresh": 0.5, "green": 0.5},
    "grass": {"green": 1}, "basil": {"green": 1}, "sage": {"green": 1},
    "rosemary": {"green": 1}, "tea": {"green": 1},
    "fig leaf": {"green": 0.7, "fruity": 0.3}, "lavender": {"green": 0.6, "floral": 0.4},
    "oakmoss": {"green": 0.5, "woody": 0.5},
    "rose": {"floral": 1}, "jasmine": {"floral": 1}, "peony": {"floral": 1},
    "magnolia": {"floral": 1}, "orange blossom": {"floral": 0.7, "citrus": 0.3},
    "peach": {"fruity": 1}, "pear": {"fruity": 1}, "apple": {"fruity": 1},
    "blackcurrant": {"fruity": 1}, "raspberry": {"fruity": 1},
    "cinnamon": {"spicy": 1}, "ginger": {"spicy": 0.7, "citrus": 0.3},
    "pink pepper": {"spicy": 1}, "black pepper": {"spicy": 1}, "cardamom": {"spicy": 1},
    "clove": {"spicy": 1}, "nutmeg": {"spicy": 1},
    "vanilla": {"gourmand": 1}, "caramel": {"gourmand": 1}, "praline": {"gourmand": 1},
    "coffee": {"gourmand": 1}, "brown sugar": {"gourmand": 1}, "buttercream": {"gourmand": 1},
    "almond": {"gourmand": 1}, "tonka": {"gourmand": 0.7, "amber": 0.3},
    "honey": {"gourmand": 0.7, "floral": 0.3},
    "cedar": {"woody": 1}, "sandalwood": {"woody": 1}, "patchouli": {"woody": 1},
    "vetiver": {"woody": 0.7, "green": 0.3},
    "pine": {"woody": 0.5, "green": 0.3, "amber": 0.2},
    "fir": {"woody": 0.5, "green": 0.3, "amber": 0.2},
    "amber": {"amber": 1}, "incense": {"amber": 1}, "labdanum": {"amber": 1},
    "benzoin": {"amber": 0.6, "gourmand": 0.4}, "tobacco": {"amber": 0.6, "woody": 0.4},
    "musk": {"musk": 1}, "white musk": {"musk": 1},
    "clean cotton": {"musk": 0.6, "fresh": 0.4}, "iris": {"musk": 0.5, "floral": 0.5},
    "ambroxan": {"musk": 0.5, "woody": 0.5},
    # Added for the real survey products (data/real/catalog.csv).
    "neroli": {"citrus": 0.5, "floral": 0.5}, "tangerine": {"citrus": 1},
    "strawberry": {"fruity": 1}, "berries": {"fruity": 1}, "tropical fruit": {"fruity": 1},
    "champagne": {"fruity": 0.6, "fresh": 0.4},
    "violet": {"floral": 1}, "gardenia": {"floral": 1}, "ambrette": {"musk": 1},
    "whipped cream": {"gourmand": 1}, "coconut": {"gourmand": 1}, "sugar": {"gourmand": 1},
    "mahogany": {"woody": 1}, "teakwood": {"woody": 1}, "oak": {"woody": 1},
    "balsam": {"amber": 0.5, "woody": 0.3, "green": 0.2},
    "eucalyptus": {"green": 0.6, "fresh": 0.4}, "spearmint": {"fresh": 0.5, "green": 0.5},
}


def to_vector(weights):
    v = np.zeros(len(NODES))
    for node, w in weights.items():
        v[NODES.index(node)] += w
    return v


def node_vector(notes, unknown=None):
    """A product's node profile: the average of its notes' node weights (sums to 1)."""
    vecs = []
    for note in notes:
        key = note.strip().lower()
        if key in NOTE_NODES:
            vecs.append(to_vector(NOTE_NODES[key]))
        elif unknown is not None:
            unknown.add(key)
    if not vecs:
        return np.zeros(len(NODES))
    v = np.sum(vecs, axis=0)
    return v / v.sum()


def quiz_matrix(items):
    """(questions, nodes) weights for the given quiz questions."""
    return np.array([to_vector(QUIZ_NODES[q]) for q in items])
