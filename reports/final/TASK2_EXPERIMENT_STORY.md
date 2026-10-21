# Task 2 experiment story

This is the presentation narrative for the frozen experiment. Exact figures come from `KEY_NUMBERS.json` and `MODEL_COMPARISON.csv`; they are not new evaluations.

## A. Dataset preparation

The research benchmark is VN-MSASar v3, filtered to Vietnamese restaurant rows: 5,869 sentence/aspect rows. The fixed seed-42 split has 4,140 train, 842 validation, and 887 test rows. It groups by review ID and normalized text, with zero exact normalized-text and review-ID overlap across splits. The sentiment labels are negative, neutral, and positive. Neutral is scarce: 208 train, 43 validation, and 43 test rows. Exact-group leakage is checked; fuzzy near-duplicates remain a limitation. The text-only sentiment protocol withholds aspect, rating, and gold label from the model. Identical text can still carry different aspect-level sentiment labels, creating irreducible ambiguity for a text-only predictor.

## B. Baselines

Rule-based sentiment provides an interpretable lower reference: test accuracy 0.5051, macro F1 0.4313. TF-IDF bigrams plus balanced logistic regression reaches accuracy 0.8095 and macro F1 0.6829. Both use the same frozen test split and text-only input. The TF-IDF model is kept for Foody enrichment because it wins this measured comparison and is small and fast; its neutral F1 is only 0.4130, so the minority class remains difficult.

## C. Proposed V1

Qwen2.5-1.5B-Instruct-GGUF Q4_K_M classifies the same text with a selected prompt and strict JSON. Its test macro F1 is 0.6218, below TF-IDF's 0.6829; neutral F1 is 0.2578. The zero-shot/few-shot prompt did not overcome class imbalance or text-only aspect ambiguity. V1 was not promoted as the Foody sentiment classifier.

## D. Proposed V2-A

V2-A tested prompt/calibration changes and produced test macro F1 0.6145, again below TF-IDF and V1. The V1 test result had already been viewed before this V2 test reuse; therefore the V2 test comparison is **exploratory** and needs untouched external confirmation. It is not evidence of a general model ranking.

## E. Aspect model

A separate text-to-aspects TF-IDF OneVsRest classifier was added because sentiment alone cannot say whether a complaint concerns food, service, price, or ambience. On its frozen aspect test, micro F1 is 0.8323 and macro F1 is 0.6336. The weaker macro result reflects uneven aspect support. Aspect prediction is not a substitute for gold aspect-conditioned sentiment.

## F. Real-world Foody inference

The source has 33,946 review rows, 31,694 AI-valid inference observations, 32,457 unique valid review documents, and 2,110 restaurant names. The sentiment and aspect models enrich the valid observations; deterministic rules extract issue hints. Foody `Classification` is not treated as trusted manually annotated gold. There is no measured Foody prediction accuracy. The real data serves the restaurant dashboard and evidence-backed retrieval demo, with a clear domain-shift caveat from VN-MSASar.

## G. RAG V1

V1 used MiniLM dense retrieval and Qwen. In the small six-query incomplete known-positive set, known-positive Hit@5 was 0, Hit@10 was 0.083, and MRR was 0.019. These are not full-corpus recall numbers. Semantic similarity alone sometimes returned evidence with the wrong complaint/aspect polarity.

## H. RAG V2

V2 added BM25, RRF hybrid candidates, bounded query expansion, and evidence rules. It blocked contradictory service praise and selected an evidence-reranked route (known-positive MRR 0.056 in the shared policy comparison). The generator remained unreliable: a food query had useful evidence but failed its answer contract after two attempts. The rejected V2 raw generations were not retained, so their exact JSON/citation failure type cannot be reconstructed.

## I. RAG V3

V3 kept the existing 32,457-document indexes and routed questions: complaint/aspect queries use filtered BM25, general queries use hybrid retrieval, and causal questions use a mart guard. Its known-positive MRR is 0.167 on the same small incomplete set, tied with plain BM25. Deterministic evidence packets precede Qwen2.5-3B Q4_K_M. A clause-level validator checks topic, polarity, review citations, frequency language, and multi-fact claims. If generation fails, a cited extractive answer preserves useful evidence; if evidence is incompatible, the system refuses. In the fixed six-case local replay: one supported 3B synthesis, two verified fallbacks, three correct insufficient-evidence responses, and **zero unsupported accepted answers**. This is demo acceptance, not statistical proof of hallucination elimination.

## J. Limits and conclusion

The retrieval labels are manually constructed and incomplete. The generation set has only six cases. A deterministic validator can reject good paraphrases and cannot replace full human review. Fallback examples may answer only part of a frequency question because one review cannot establish prevalence. Sparse restaurant time series cannot establish why a rating changed. The V2-A test hygiene limitation and Foody gold-label gap remain. The engineering conclusion is narrow: TF-IDF is the measured sentiment choice; aspect inference and evidence-first RAG V3 make the local demo inspectable and conservative.
