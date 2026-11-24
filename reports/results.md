# Model comparison: `data/synthetic`

## Data

- 120 respondents; 1015 product ratings (median 8 per person) on 16 survey products; catalog of 40 products.

- 6 everyday-smell questions; 52% named at least one product they already love (optional input).

- Split by person: 96 train / 24 test (seed 0). Hyperparameters are chosen by cross-validation inside train; test is scored once.

## Results on held-out people

MAE = average miss in rating points (1-5 scale). liked@1 / liked@3 = share of the model's top 1 / top 3 picks, among products the person rated, that they rated 4 or 5. Brackets are 95% paired bootstrap intervals; an interval that crosses 0 means the difference is not reliable at this sample size.

| model | MAE (pts) | liked@1 | liked@3 | MAE vs baseline | liked@3 vs baseline |
|---|---|---|---|---|---|
| Popularity (baseline) | 0.795 | 0.625 | 0.514 | - | - |
| Note match | 0.676 | 0.792 | 0.625 | -0.118 [-0.175, -0.062] | +0.111 [+0.000, +0.236] |
| Persona clusters | 0.709 | 0.667 | 0.597 | -0.086 [-0.113, -0.060] | +0.083 [+0.028, +0.153] |
| Persona clusters + notes | 0.658 | 0.833 | 0.611 | -0.137 [-0.190, -0.087] | +0.097 [-0.014, +0.222] |

## Same data, repeated splits

20 different random person splits of this same dataset.

| model | mean MAE | MAE beat baseline | mean liked@3 | liked@3 beat baseline |
|---|---|---|---|---|
| Note match | 0.706 | 20/20 | 0.565 | 20/20 |
| Persona clusters | 0.748 | 19/20 | 0.527 | 19/20 |
| Persona clusters + notes | 0.711 | 20/20 | 0.549 | 20/20 |
| Popularity (baseline) | 0.787 | - | 0.479 | - |

## Personas found by k-means

k = 5 personas (shrink = 5), chosen by 5-fold cross-validation on training people.

Silhouette = 0.23 (0 = overlapping groups, 1 = cleanly separated).

Agreement with the hidden synthetic personas (adjusted Rand index) = 0.09 (1 = perfect).

| persona | people | leans toward | leans away from | top survey products |
|---|---|---|---|---|
| 0 | 10 | floral, gourmand | spicy, fresh | Sweet Vanilla No. 1; Clean Linen No. 1; Fresh Aquatic No. 1 |
| 1 | 19 | floral, fresh | gourmand, spicy | Fresh Aquatic No. 1; Green Fougere No. 1; Clean Linen No. 1 |
| 2 | 19 | gourmand, spicy | floral, fresh | Sweet Vanilla No. 1; Vanilla Amber No. 1; Bakery No. 1 |
| 3 | 21 | fresh, woody | floral, gourmand | Fresh Aquatic No. 1; Rain & Moss No. 1; Citrus Grove No. 1 |
| 4 | 27 | gourmand, floral | - | Fresh Aquatic No. 1; Citrus Grove No. 1; Clean Linen No. 1 |

## How many responses do we need?

Same hidden world, a fresh sample of people for each size and seed (10 seeds), 20% of people held out each time.

**Mean MAE (lower is better)**

| respondents | Note match | Persona clusters | Persona clusters + notes | Popularity (baseline) |
|---|---|---|---|---|
| 30 | 0.681 | 0.739 | 0.688 | 0.743 |
| 60 | 0.728 | 0.794 | 0.736 | 0.798 |
| 120 | 0.707 | 0.749 | 0.712 | 0.771 |
| 250 | 0.706 | 0.734 | 0.710 | 0.766 |
| 500 | 0.704 | 0.735 | 0.714 | 0.762 |

**Mean liked@3 (higher is better)**

| respondents | Note match | Persona clusters | Persona clusters + notes | Popularity (baseline) |
|---|---|---|---|---|
| 30 | 0.528 | 0.417 | 0.506 | 0.406 |
| 60 | 0.486 | 0.447 | 0.472 | 0.442 |
| 120 | 0.533 | 0.496 | 0.533 | 0.453 |
| 250 | 0.519 | 0.490 | 0.508 | 0.442 |
| 500 | 0.500 | 0.468 | 0.492 | 0.438 |

**Seeds where the model's MAE beat the baseline's**

| respondents | Note match | Persona clusters | Persona clusters + notes |
|---|---|---|---|
| 30 | 8/10 | 5/10 | 7/10 |
| 60 | 10/10 | 4/10 | 9/10 |
| 120 | 10/10 | 9/10 | 10/10 |
| 250 | 10/10 | 10/10 | 10/10 |
| 500 | 10/10 | 10/10 | 10/10 |
