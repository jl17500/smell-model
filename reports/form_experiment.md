# Survey design experiment

120 simulated respondents per survey, 10 samples of people, the same hidden world. Within a sample the people, their tastes and their product ratings are identical across designs; only the questions asked change. Every design is scored on the same held-out ratings (products a person names as loved are never scored). 'Smells only' ignores loved products entirely, which is how the app works for someone who names none.

## Mean MAE (lower is better)

| design | Popularity | Note match: smells only | Note match: + loved | Persona+notes: smells only | Persona+notes: + loved |
|---|---|---|---|---|---|
| A. Current: 6 smells, 4 choices, 1 loved product | 0.767 | 0.711 | 0.709 | 0.715 | 0.713 |
| B. A + peaches, clean laundry, orange peel | 0.767 | 0.704 | 0.701 | 0.709 | 0.701 |
| C. B on a 1-6 scale (no neutral) | 0.767 | 0.699 | 0.698 | 0.701 | 0.697 |
| D. C + up to 3 loved products | 0.767 | 0.699 | 0.702 | 0.701 | 0.696 |

## Mean liked@3 (higher is better)

| design | Popularity | Note match: smells only | Note match: + loved | Persona+notes: smells only | Persona+notes: + loved |
|---|---|---|---|---|---|
| A. Current: 6 smells, 4 choices, 1 loved product | 0.446 | 0.518 | 0.525 | 0.515 | 0.525 |
| B. A + peaches, clean laundry, orange peel | 0.446 | 0.524 | 0.533 | 0.518 | 0.529 |
| C. B on a 1-6 scale (no neutral) | 0.446 | 0.528 | 0.542 | 0.522 | 0.533 |
| D. C + up to 3 loved products | 0.446 | 0.528 | 0.531 | 0.522 | 0.529 |

## Samples where MAE beat design A (same people, same model)

| design | Popularity | Note match: smells only | Note match: + loved | Persona+notes: smells only | Persona+notes: + loved |
|---|---|---|---|---|---|
| A. Current: 6 smells, 4 choices, 1 loved product | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| B. A + peaches, clean laundry, orange peel | 0/10 | 9/10 | 9/10 | 7/10 | 8/10 |
| C. B on a 1-6 scale (no neutral) | 0/10 | 8/10 | 8/10 | 8/10 | 8/10 |
| D. C + up to 3 loved products | 0/10 | 8/10 | 7/10 | 8/10 | 8/10 |
