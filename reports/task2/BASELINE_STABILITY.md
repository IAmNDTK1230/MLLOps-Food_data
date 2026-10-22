# Baseline stability on the frozen split

Split SHA256: `860387be19c64220f91cc2b5e764aa3c99aa18daba9c776fde38e05da30a5ff9`. Train/validation/test memberships are unchanged. Each seed trains the accepted TF-IDF + Logistic Regression configuration from scratch on the same 4,140 train rows; the accepted serialized model is not touched.

`lbfgs` may be deterministic here. Identical scores are reported with standard deviation zero. The repeated test analysis is exploratory because the accepted test results were already viewed; it is not a new held-out model-selection claim.

| Split | Accuracy mean ± sample SD | Macro F1 mean ± sample SD | Weighted F1 mean ± sample SD | Neutral F1 mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.819477 ± 0.000000 | 0.709242 ± 0.000000 | 0.821549 ± 0.000000 | 0.471910 ± 0.000000 |
| test | 0.809470 ± 0.000000 | 0.682854 ± 0.000000 | 0.812104 ± 0.000000 | 0.413043 ± 0.000000 |

Individual runs and per-class F1 are in `baseline_three_seed.csv`.

LinearSVC is a separate classical comparison, not a replacement baseline. C=0.5 was selected on validation macro F1 (0.701795); its single test macro F1 is 0.667210. See `linearsvc_validation_selection.csv` and `classical_model_comparison.csv`. No test-based tuning was performed.
