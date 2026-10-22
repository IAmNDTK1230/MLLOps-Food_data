# Evidence-based gap analysis

No directly comparable paper result exists, so a numeric paper-group gap is unavailable. Within the frozen group experiment, accepted TF-IDF macro F1 0.682854 exceeds Qwen V1 0.621789. Neutral F1 is 0.413043 versus 0.257778. The neutral class has only 294/5,869 rows and its validation F1 rises from 0.170213 to 0.471910 with balanced weights: measured evidence that weighting helps this class. Eighty-one normalized texts have conflicting aspect-level labels globally, including eleven in test: a text-only task has inherent ambiguity.

Potential causes not isolated experimentally include domain differences, preprocessing, hyperparameters, pretrained model data, and task/input differences. These are hypotheses, not proven explanations of the measured gap. Aspect-conditioned V2-A is exploratory because its test set was previously inspected; RAG V3 addresses evidence grounding for the demo rather than sentiment benchmark macro F1.
