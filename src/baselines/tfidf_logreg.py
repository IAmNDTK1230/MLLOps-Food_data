from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ml.features import normalize_text

EXPERIMENTS = (
    {"name": "A_unigram_c1", "ngram_range": (1, 1), "c": 1.0},
    {"name": "B_bigram_c1", "ngram_range": (1, 2), "c": 1.0},
    {"name": "C_bigram_c2", "ngram_range": (1, 2), "c": 2.0},
)


def select_validation_trial(experiments):
    """Stable candidate-order tie break; only validation macro F1 is consulted."""
    return max(range(len(experiments)), key=lambda index: experiments[index]["macro_f1"])


def build_pipeline(experiment, class_weight="balanced", seed=42):
    """Only text enters this pipeline; vocabulary/IDF/classifier fit on train."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=False,
                    ngram_range=tuple(experiment["ngram_range"]),
                    max_features=20000,
                    min_df=2,
                    token_pattern=r"(?u)\b\w+\b",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=experiment["c"],
                    max_iter=1500,
                    class_weight=class_weight,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
