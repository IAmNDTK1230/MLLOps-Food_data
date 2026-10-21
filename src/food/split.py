import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.food.data import load_prepared
from src.ml.features import normalize_text
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum

NAMES = ("train", "validation", "test")


def leakage_groups(frame):
    """Transitive closure of shared review IDs and normalized text (case preserved)."""
    parents, seen = list(range(len(frame))), {}

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, row in enumerate(frame.to_dict("records")):
        for key in (("review", str(row["review_id"])), ("text", normalize_text(row["text"]))):
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    return np.array([find(index) for index in range(len(frame))])


def assign_splits(frame, config):
    frame = frame.reset_index(drop=True)
    groups = leakage_groups(frame)
    if len(set(groups)) < 3:
        raise ValueError("Need at least three independent review/text groups")
    target = np.array([config.train_fraction, config.validation_fraction, config.test_fraction])
    population = frame.sentiment.value_counts(normalize=True)
    best, best_score = None, float("inf")
    # Label-aware deterministic group search; no estimator or test predictions involved.
    first = GroupShuffleSplit(
        n_splits=64, test_size=1 - config.train_fraction, random_state=config.random_seed
    )
    for train, holdout in first.split(frame, groups=groups):
        if len(set(groups[holdout])) < 2:
            continue
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=config.test_fraction / (config.validation_fraction + config.test_fraction),
            random_state=config.random_seed + 1,
        )
        val, test = next(second.split(frame.iloc[holdout], groups=groups[holdout]))
        partitions = (train, holdout[val], holdout[test])
        if any(set(frame.iloc[ids].sentiment) != set(population.index) for ids in partitions):
            continue
        score = float(abs(np.array([len(ids) for ids in partitions]) / len(frame) - target).sum())
        score += sum(
            float(
                (
                    frame.iloc[ids]
                    .sentiment.value_counts(normalize=True)
                    .reindex(population.index, fill_value=0)
                    - population
                )
                .abs()
                .sum()
            )
            for ids in partitions
        )
        if score < best_score:
            best, best_score = partitions, score
    if best is None:
        raise ValueError(
            "Cannot form three leakage-free splits with every class; need more independent groups"
        )
    result = frame.copy()
    for name, indices in zip(NAMES, best, strict=True):
        result.loc[indices, "split"] = name
    metadata = {
        "strategy": "review_id_and_normalized_text_connected_components",
        "text_grouping": "NFC + whitespace normalization, case and accents preserved",
        "random_seed": config.random_seed,
        "group_count": len(set(groups)),
        "requested_fractions": dict(zip(NAMES, target.tolist(), strict=True)),
        "actual_fractions": result.split.value_counts(normalize=True).to_dict(),
        "counts": result.split.value_counts().to_dict(),
        "unique_reviews": {
            name: int(result.loc[result.split == name].review_id.nunique()) for name in NAMES
        },
        "class_distributions": {
            name: result.loc[result.split == name].sentiment.value_counts().to_dict()
            for name in NAMES
        },
        "class_percentages": {
            name: (
                result.loc[result.split == name].sentiment.value_counts(normalize=True) * 100
            ).to_dict()
            for name in NAMES
        },
        "shared_by": ["baseline_rule", "baseline_tfidf", "proposed_v1"],
        "leakage_checks": {
            "review_ids_spanning_splits": int(
                result.groupby("review_id").split.nunique().gt(1).sum()
            ),
            "normalized_texts_spanning_splits": int(
                result.groupby("text").split.nunique().gt(1).sum()
            ),
        },
        "source_splits": "Retained as original_split; recombined for requested canonical 70/15/15 experiment",
        "near_duplicate_limitation": "Exact normalized text protected; no additional fuzzy duplicate guarantee",
    }
    return result, metadata


def split_dataset(config):
    frame, preparation = load_prepared(config)
    rows, metadata = assign_splits(frame, config)
    path = config.data_dir / "splits/reviews.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)
    metadata.update(
        processed_sha256=preparation["processed_sha256"], split_file_sha256=file_checksum(path)
    )
    atomic_json(config.data_dir / "splits/metadata.json", metadata)
    return metadata


def load_splits(config):
    _, preparation = load_prepared(config)
    path = config.data_dir / "splits/reviews.parquet"
    meta_path = config.data_dir / "splits/metadata.json"
    if not path.exists() or not meta_path.exists():
        raise ValueError("Canonical splits missing; run food-split")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = dict(
        zip(
            NAMES,
            [config.train_fraction, config.validation_fraction, config.test_fraction],
            strict=True,
        )
    )
    if (
        metadata["processed_sha256"] != preparation["processed_sha256"]
        or metadata["split_file_sha256"] != file_checksum(path)
        or metadata["random_seed"] != config.random_seed
        or metadata["requested_fractions"] != expected
    ):
        raise ValueError("Saved splits/data/config changed; rerun food-split")
    frame = pd.read_parquet(path)
    if (
        set(frame.split) != set(NAMES)
        or pd.DataFrame({"group": leakage_groups(frame), "split": frame.split})
        .groupby("group")
        .split.nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("Invalid saved splits or review/text leakage")
    return frame, metadata, preparation
