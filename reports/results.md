# Model comparison: `data/synthetic`

## Data

- 120 respondents; 1015 product ratings (median 8 per person) on 16 survey products; catalog of 40 products.

- 9 everyday-smell questions; 52% named at least one product they already love (optional input).

- Split by person: 96 train / 24 test (seed 0). Hyperparameters are chosen by cross-validation inside train; test is scored once.

## Results on held-out people

MAE = average miss in rating points (1-5 scale). liked@1 / liked@3 = share of the model's top 1 / top 3 picks, among products the person rated, that they rated 4 or 5. Brackets are 95% paired bootstrap intervals; an interval that crosses 0 means the difference is not reliable at this sample size.

| model | MAE (pts) | liked@1 | liked@3 | MAE vs baseline | liked@3 vs baseline |
|---|---|---|---|---|---|
| Popularity (baseline) | 0.790 | 0.625 | 0.500 | - | - |
| Note match | 0.674 | 0.792 | 0.597 | -0.116 [-0.169, -0.067] | +0.097 [+0.014, +0.181] |
| Persona clusters | 0.753 | 0.542 | 0.583 | -0.037 [-0.087, +0.014] | +0.083 [+0.014, +0.153] |
| Persona clusters + notes | 0.681 | 0.750 | 0.597 | -0.109 [-0.170, -0.049] | +0.097 [+0.014, +0.181] |

## Same data, repeated splits

20 different random person splits of this same dataset. loved hit@10 is a second check that uses only the optional loved-products question: hide each loved product, rank all 40 catalog products from smell answers alone, and count how often it lands in the top 10 (random guessing: 0.25).

| model | mean MAE | MAE beat baseline | mean liked@3 | liked@3 beat baseline | loved hit@10 |
|---|---|---|---|---|---|
| Note match | 0.706 | 20/20 | 0.556 | 20/20 | 0.490 |
| Persona clusters | 0.744 | 19/20 | 0.528 | 19/20 | 0.224 |
| Persona clusters + notes | 0.701 | 20/20 | 0.560 | 20/20 | 0.479 |
| Popularity (baseline) | 0.785 | - | 0.472 | - | 0.249 |

## Personas found by k-means

k = 4 personas (shrink = 5), chosen by 5-fold cross-validation on training people.

Silhouette = 0.19 (0 = overlapping groups, 1 = cleanly separated).

Agreement with the hidden synthetic personas (adjusted Rand index) = 0.11 (1 = perfect).

| persona | people | leans toward | leans away from | top survey products |
|---|---|---|---|---|
| 0 | 31 | citrus, fruity | gourmand, amber | Fresh Aquatic No. 1; Green Fougere No. 1; Clean Linen No. 1 |
| 1 | 14 | gourmand, spicy | fresh, green | Sweet Vanilla No. 1; Bakery No. 1; Spiced Chai No. 1 |
| 2 | 20 | gourmand, amber | citrus, fruity | Green Fougere No. 1; Fresh Aquatic No. 1; Rain & Moss No. 1 |
| 3 | 31 | gourmand, floral | - | Fresh Aquatic No. 1; Clean Linen No. 1; Citrus Grove No. 1 |

## How many responses do we need?

Same hidden world, a fresh sample of people for each size and seed (10 seeds), 20% of people held out each time.

**Mean MAE (lower is better)**

| respondents | Note match | Persona clusters | Persona clusters + notes | Popularity (baseline) |
|---|---|---|---|---|
| 30 | 0.670 | 0.735 | 0.683 | 0.743 |
| 60 | 0.726 | 0.776 | 0.730 | 0.798 |
| 120 | 0.702 | 0.734 | 0.696 | 0.767 |
| 250 | 0.694 | 0.727 | 0.694 | 0.764 |
| 500 | 0.700 | 0.719 | 0.699 | 0.761 |

**Mean liked@3 (higher is better)**

| respondents | Note match | Persona clusters | Persona clusters + notes | Popularity (baseline) |
|---|---|---|---|---|
| 30 | 0.494 | 0.417 | 0.456 | 0.400 |
| 60 | 0.503 | 0.469 | 0.500 | 0.442 |
| 120 | 0.531 | 0.503 | 0.529 | 0.446 |
| 250 | 0.536 | 0.496 | 0.535 | 0.435 |
| 500 | 0.500 | 0.481 | 0.499 | 0.431 |

**Seeds where the model's MAE beat the baseline's**

| respondents | Note match | Persona clusters | Persona clusters + notes |
|---|---|---|---|
| 30 | 9/10 | 6/10 | 8/10 |
| 60 | 10/10 | 8/10 | 10/10 |
| 120 | 10/10 | 9/10 | 10/10 |
| 250 | 10/10 | 9/10 | 10/10 |
| 500 | 10/10 | 10/10 | 10/10 |
