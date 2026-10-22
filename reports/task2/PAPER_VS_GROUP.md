# Paper vs Group — source audit

The three local baselines were **not reconstructed from a cited published classifier paper**. The rule lexicon was authored in `src/baselines/rule_based.py`; the two linear models were implemented with scikit-learn. Their scores are original group measurements, not replications of published scores. The [VN-MSASar v3 dataset card](https://huggingface.co/datasets/phamluan/vn_msa_sar_v3) is the dataset source, not a paper reporting these three baselines on the group's filtered split. The project README, experiment notes, reports and source contain no citation to a matching baseline paper or verified paper macro F1.

| Baseline | Paper reconstruction | Group test macro F1 | Comparison |
| --- | --- | ---: | --- |
| Rule-based sentiment | No; manually authored domain lexicon | 0.431279 | NOT DIRECTLY COMPARABLE |
| TF-IDF + Logistic Regression | No; local scikit-learn pipeline | 0.682854 | NOT DIRECTLY COMPARABLE |
| TF-IDF + LinearSVC | No; extra validation-selected local comparison | 0.667210 | NOT DIRECTLY COMPARABLE |

Method references: [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), and [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) describe algorithms/APIs; they supply no matching performance claim.

**Dataset gap:** the public card covers a multilingual hotel/restaurant dataset, while the group filtered Vietnamese restaurant rows and fixed its own split. **Label/task gap:** source rows are aspect-level sentiment, while these three classifiers receive text only and collapse sentiment to three classes. **Preprocessing gap:** no cited paper specifies the same normalization, TF-IDF, grouping and train-only fitting. **Metric gap:** there is no published macro F1 on the exact 887-row held-out set. Therefore every numerical absolute paper-group gap remains blank in `paper_vs_group.csv`; no paper score is invented.
