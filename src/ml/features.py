from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if TYPE_CHECKING:
    from src.ml.config import Experiment


def normalize_text(text: str) -> str:
    # Preserve Vietnamese accents; NFC joins equivalent decomposed characters.
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pipeline(experiment: Experiment, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=False,
                    max_features=experiment.max_features,
                    ngram_range=experiment.ngram_range,
                    token_pattern=r"(?u)\b\w+\b",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=experiment.c,
                    max_iter=experiment.max_iter,
                    class_weight="balanced",
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
