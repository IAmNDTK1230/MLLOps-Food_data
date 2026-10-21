import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

LABELS = ("negative", "neutral", "positive")


class Experiment(BaseModel):
    name: str
    max_features: int = Field(gt=0)
    ngram_range: tuple[int, int] = (1, 1)
    c: float = Field(default=1, gt=0)
    max_iter: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def valid_ngrams(self):
        if not 1 <= self.ngram_range[0] <= self.ngram_range[1]:
            raise ValueError("ngram_range must be ordered positive integers")
        return self


class MLConfig(BaseModel):
    data_dir: Path = Path("data/ml")
    reports_dir: Path = Path("reports")
    models_dir: Path = Path("models")
    random_seed: int = 42
    train_fraction: float = Field(default=0.7, gt=0, lt=1)
    validation_fraction: float = Field(default=0.15, gt=0, lt=1)
    test_fraction: float = Field(default=0.15, gt=0, lt=1)
    group_by_product: bool = True
    label_rules: dict[int, str] = {
        1: "negative",
        2: "negative",
        3: "neutral",
        4: "positive",
        5: "positive",
    }
    experiments: list[Experiment] = [
        Experiment(name="A_unigram", max_features=10000),
        Experiment(name="B_bigram", max_features=10000, ngram_range=(1, 2)),
        Experiment(name="C_bigram_c2", max_features=20000, ngram_range=(1, 2), c=2),
    ]

    @model_validator(mode="after")
    def valid_config(self):
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1) > 1e-9:
            raise ValueError("Split fractions must sum to 1")
        if set(self.label_rules) != {1, 2, 3, 4, 5} or not set(self.label_rules.values()).issubset(
            LABELS
        ):
            raise ValueError("Label rules must map ratings 1–5 to supported sentiment labels")
        if len(self.experiments) < 3 or len({e.name for e in self.experiments}) != len(
            self.experiments
        ):
            raise ValueError("Configure at least three uniquely named experiments")
        return self

    @classmethod
    def load(cls, path: str = "params.yaml"):
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        config = cls.model_validate(payload["ml"])
        return config


class TrackingConfig(BaseModel):
    tracking_uri: str = "sqlite:///state/mlflow.db"
    experiment_name: str = "/Shared/shopee-sentiment"
    registry_uri: str = "databricks-uc"

    @model_validator(mode="after")
    def no_embedded_credentials(self):
        parsed = urlsplit(self.tracking_uri)
        if parsed.username or parsed.password or parsed.query:
            raise ValueError(
                "Use MLflow authentication environment variables, never credentials in the tracking URI"
            )
        return self

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///state/mlflow.db",
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME") or "/Shared/shopee-sentiment",
            registry_uri=os.getenv("MLFLOW_REGISTRY_URI") or "databricks-uc",
        )
