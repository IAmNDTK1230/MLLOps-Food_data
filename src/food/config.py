from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class FoodConfig(BaseModel):
    data_dir: Path = Path("data/food")
    reports_dir: Path = Path("reports")
    dataset_repo: str = "phamluan/vn_msa_sar_v3"
    dataset_revision: str = "6f3f17b390dcf1bf411804e1bc8246bf88ea1bf4"
    random_seed: int = 42
    train_fraction: float = Field(default=0.7, gt=0, lt=1)
    validation_fraction: float = Field(default=0.15, gt=0, lt=1)
    test_fraction: float = Field(default=0.15, gt=0, lt=1)
    tokenizer_repo: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer_revision: str = "a09a35458c702b33eeacc393d103063234e8bc28"
    chunk_sizes: list[int] = [128, 256, 512]
    overlaps: list[int] = [0, 32, 64]

    @model_validator(mode="after")
    def check(self):
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1) > 1e-9:
            raise ValueError("Split fractions must sum to one")
        if (
            not self.chunk_sizes
            or not self.overlaps
            or any(
                size <= 0 or overlap < 0 or overlap >= size
                for size in self.chunk_sizes
                for overlap in self.overlaps
            )
        ):
            raise ValueError("Chunk sizes must be positive and each overlap smaller than each size")
        return self

    @classmethod
    def load(cls, path="params.yaml"):
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["food"])
