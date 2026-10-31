# Test vs validation performance check

Model: `random_forest_export:v12` (alias `prod`), trained in run `celestial-sky-17`, tested in run `cryptic-fog-31`

| Metric | Validation | Test | Difference | Tolerance | Result |
|---|---|---|---|---|---|
| MAE | 33.80 | 31.63 | -6.4% | <= +10% | OK |
| r2 | 0.565 | 0.624 | +0.058 | drop <= 0.05 | OK |

**PASSED: no overfitting detected**
