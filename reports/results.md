# Model comparison: `data/synthetic`

## Data

- 120 respondents; 1074 product ratings (median 9 per person) on 16 survey products; catalog of 40 products.

- 57% named a favorite product found in the catalog.

- Split by person: 96 train / 24 test (seed 0). Hyperparameters are chosen by cross-validation inside train; test is scored once.

## Results on held-out people

MAE = average miss in rating points (1-5 scale). liked@1 / liked@3 = share of the model's top 1 / top 3 picks, among products the person rated, that they rated 4 or 5. Brackets are 95% paired bootstrap intervals; an interval that crosses 0 means the difference is not reliable at this sample size.

| model | MAE (pts) | liked@1 | liked@3 | MAE vs baseline | liked@3 vs baseline |
|---|---|---|---|---|---|
| Popularity (baseline) | 0.747 | 0.625 | 0.653 | - | - |
| Note match | 0.725 | 0.583 | 0.653 | -0.021 [-0.079, +0.035] | -0.000 [-0.056, +0.056] |
| Persona clusters | 0.754 | 0.583 | 0.653 | +0.007 [-0.027, +0.042] | +0.000 [-0.056, +0.056] |
| Persona clusters + notes | 0.729 | 0.625 | 0.597 | -0.017 [-0.072, +0.037] | -0.056 [-0.125, +0.014] |

## Same data, repeated splits

20 different random person splits of this same dataset.

| model | mean MAE | MAE beat baseline | mean liked@3 | liked@3 beat baseline |
|---|---|---|---|---|
| Note match | 0.721 | 20/20 | 0.623 | 17/20 |
| Persona clusters | 0.760 | 13/20 | 0.584 | 6/20 |
| Persona clusters + notes | 0.727 | 19/20 | 0.606 | 11/20 |
| Popularity (baseline) | 0.768 | - | 0.588 | - |

## Personas found by k-means

k = 2 personas (shrink = 5), chosen by 5-fold cross-validation on training people.

Silhouette = 0.26 (0 = overlapping groups, 1 = cleanly separated).

Agreement with the hidden synthetic personas (adjusted Rand index) = 0.18 (1 = perfect).

| persona | people | leans toward | leans away from | top survey products |
|---|---|---|---|---|
| 0 | 43 | gourmand, spicy | - | Bakery No. 1; Fresh Aquatic No. 1; Christmas Forest No. 1 |
| 1 | 53 | fresh, citrus | gourmand, spicy | Fresh Aquatic No. 1; Citrus Grove No. 1; Clean Linen No. 1 |

## How many responses do we need?

Same hidden world, a fresh sample of people for each size and seed (10 seeds), 20% of people held out each time.

**Mean MAE (lower is better)**

| respondents | Note match | Persona clusters | Persona clusters + notes | Popularity (baseline) |
|---|---|---|---|---|
| 30 | 0.764 | 0.804 | 0.774 | 0.797 |
| 60 | 0.734 | 0.777 | 0.740 | 0.793 |
| 120 | 0.717 | 0.760 | 0.723 | 0.777 |
| 250 | 0.691 | 0.724 | 0.695 | 0.753 |
| 500 | 0.713 | 0.730 | 0.711 | 0.770 |

**Mean liked@3 (higher is better)**

| respondents | Note match | Persona clusters | Persona clusters + notes | Popularity (baseline) |
|---|---|---|---|---|
| 30 | 0.567 | 0.511 | 0.517 | 0.511 |
| 60 | 0.519 | 0.483 | 0.506 | 0.461 |
| 120 | 0.581 | 0.540 | 0.576 | 0.514 |
| 250 | 0.591 | 0.565 | 0.588 | 0.517 |
| 500 | 0.579 | 0.554 | 0.575 | 0.521 |

**Seeds where the model's MAE beat the baseline's**

| respondents | Note match | Persona clusters | Persona clusters + notes |
|---|---|---|---|
| 30 | 8/10 | 4/10 | 6/10 |
| 60 | 9/10 | 9/10 | 9/10 |
| 120 | 10/10 | 9/10 | 10/10 |
| 250 | 10/10 | 9/10 | 10/10 |
| 500 | 10/10 | 10/10 | 10/10 |
