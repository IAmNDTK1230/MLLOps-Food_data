import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.ml.config import MLConfig
from src.ml.data import dataset_hash, require_trainable
from src.ml.features import normalize_text
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def leakage_groups(frame: pd.DataFrame, group_products: bool) -> np.ndarray:
    """Connected components join matching comments and, optionally, product IDs."""
    parents = list(range(len(frame)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(frame.to_dict("records")):
        keys = [("text", normalize_text(row["comment"]).casefold())]
        if group_products and pd.notna(row.get("product_id")) and row.get("product_id"):
            keys.append(("product", str(row["product_id"])))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    return np.array([find(index) for index in range(len(frame))])


def assign_splits(frame: pd.DataFrame, config: MLConfig) -> tuple[pd.DataFrame, dict]:
    groups = leakage_groups(frame, config.group_by_product)
    if len(set(groups)) < 3:
        raise ValueError(
            "Need at least three independent product/text groups for train/validation/test"
        )
    labels = set(frame["sentiment"])
    target = np.array([config.train_fraction, config.validation_fraction, config.test_fraction])
    best, best_score = None, float("inf")
    # Deterministic search for balanced *group* assignments, never using model performance.
    splitter = GroupShuffleSplit(
        n_splits=64, test_size=1 - config.train_fraction, random_state=config.random_seed
    )
    for train, holdout in splitter.split(frame, frame["sentiment"], groups):
        if len(set(groups[holdout])) < 2:
            continue
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=config.test_fraction / (config.validation_fraction + config.test_fraction),
            random_state=config.random_seed + 1,
        )
        validation_idx, test_idx = next(second.split(frame.iloc[holdout], groups=groups[holdout]))
        partitions = [train, holdout[validation_idx], holdout[test_idx]]
        if any(set(frame.iloc[indices]["sentiment"]) != labels for indices in partitions):
            continue
        sizes = np.array([len(indices) for indices in partitions]) / len(frame)
        score = float(abs(sizes - target).sum())
        population = frame["sentiment"].value_counts(normalize=True)
        score += sum(
            float(
                (
                    frame.iloc[indices]["sentiment"]
                    .value_counts(normalize=True)
                    .reindex(population.index, fill_value=0)
                    - population
                )
                .abs()
                .sum()
            )
            for indices in partitions
        )
        if score < best_score:
            best, best_score = partitions, score
    if best is None:
        raise ValueError(
            "Cannot form leakage-free splits containing every sentiment class. Supply more independent reviews/products per class."
        )
    result = frame.copy()
    result["split"] = ""
    for indices, name in zip(best, ("train", "validation", "test"), strict=True):
        result.loc[result.index[indices], "split"] = name
    metadata = {
        "random_seed": config.random_seed,
        "strategy": "product_and_comment_connected_groups"
        if config.group_by_product
        else "comment_connected_groups",
        "group_count": len(set(groups)),
        "requested_fractions": dict(
            zip(("train", "validation", "test"), target.tolist(), strict=True)
        ),
        "counts": result["split"].value_counts().to_dict(),
        "labels": {
            name: result[result["split"] == name]["sentiment"].value_counts().to_dict()
            for name in ("train", "validation", "test")
        },
        "dataset_hash": dataset_hash(frame),
        "group_by_product": config.group_by_product,
    }
    return result, metadata


def split_dataset(config: MLConfig) -> dict:
    frame, _ = require_trainable(config)
    result, metadata = assign_splits(frame, config)
    path = config.data_dir / "splits/reviews.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    metadata["split_file_sha256"] = file_checksum(path)
    atomic_json(config.data_dir / "splits/metadata.json", metadata)
    return metadata


def load_splits(config: MLConfig) -> tuple[pd.DataFrame, dict, dict]:
    _, report = require_trainable(config)
    path = config.data_dir / "splits/reviews.parquet"
    meta_path = config.data_dir / "splits/metadata.json"
    if not path.exists() or not meta_path.exists():
        raise ValueError("ML splits missing; run ml-split")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata["dataset_hash"] != report["dataset_hash"] or metadata[
        "split_file_sha256"
    ] != file_checksum(path):
        raise ValueError("Split dataset changed; rerun ml-split")
    if (
        metadata["random_seed"] != config.random_seed
        or metadata["group_by_product"] != config.group_by_product
    ):
        raise ValueError("Split configuration changed; rerun ml-split")
    expected = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "test": config.test_fraction,
    }
    if metadata["requested_fractions"] != expected:
        raise ValueError("Split fractions changed; rerun ml-split")
    frame = pd.read_parquet(path)
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ValueError("Invalid split assignments")
    # Validate saved membership too; never trust edited metadata alone.
    groups = leakage_groups(frame, config.group_by_product)
    if (
        pd.DataFrame({"group": groups, "split": frame["split"]})
        .groupby("group")["split"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("Saved splits contain product/comment leakage")
    return frame, metadata, report
