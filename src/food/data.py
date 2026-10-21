import json
import re
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.food.adapters import LABEL_MAPPING, LICENSE_URL, REQUIRED, SOURCE_URL, adapt_vn_msasar
from src.storage.manifest import atomic_json
from src.utils.hashing import file_checksum


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def hub_metadata(repo, revision, dataset=True):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or not re.fullmatch(r"[\w.-]+", revision):
        raise ValueError("Invalid public Hugging Face repository/revision")
    kind = "datasets" if dataset else "models"
    return json.loads(fetch(f"https://huggingface.co/api/{kind}/{repo}/revision/{revision}"))


def read_source(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError("Source must be CSV, JSONL or Parquet")


def import_dataset(config, dataset="vn_msasar_v3", input_path=None):
    if dataset != "vn_msasar_v3":
        raise ValueError("Only VN-MSASar v3 is primary; external datasets cannot be merged here")
    root = config.data_dir / "source/vn_msasar_v3"
    root.mkdir(parents=True, exist_ok=True)
    files, revision = [], config.dataset_revision
    source_url = f"https://huggingface.co/datasets/{config.dataset_repo}"
    license_name, method = "cc-by-4.0", "manual_import_from_documented_v3_source"
    card_path = root / "README.md"
    if input_path:
        input_path = Path(input_path)
        paths = (
            sorted(p for p in input_path.rglob("*") if p.suffix in {".csv", ".parquet", ".jsonl"})
            if input_path.is_dir()
            else [input_path]
        )
        if not paths or any(not path.is_file() for path in paths):
            raise ValueError("No source files found in --input")
        if len({path.name for path in paths}) != len(paths):
            raise ValueError("Source files need unique basenames; use the original split filenames")
        for path in paths:
            destination = root / path.name
            if destination.exists() and destination.resolve() != path.resolve():
                # An explicit fresh import replaces only its matching source snapshot.
                if file_checksum(destination) != file_checksum(path):
                    shutil.copyfile(path, destination)
            elif destination.resolve() != path.resolve():
                shutil.copyfile(path, destination)
            files.append(destination)
        supplied_card = (
            input_path / "README.md" if input_path.is_dir() else input_path.parent / "README.md"
        )
        if supplied_card.exists() and supplied_card.resolve() != card_path.resolve():
            shutil.copyfile(supplied_card, card_path)
    else:
        try:
            metadata = hub_metadata(config.dataset_repo, revision)
            revision = metadata["sha"]
            license_name = metadata.get("cardData", {}).get("license")
            if not license_name:
                raise ValueError("Dataset license absent; import requires attribution")
            base = f"https://huggingface.co/datasets/{config.dataset_repo}/resolve/{revision}"
            card_path.write_bytes(fetch(f"{base}/README.md"))
            for sibling in metadata["siblings"]:
                name = sibling["rfilename"]
                if name.endswith(".parquet"):
                    destination = root / Path(name).name
                    temporary = destination.with_suffix(".download")
                    temporary.write_bytes(fetch(f"{base}/{name}"))
                    temporary.replace(destination)
                    files.append(destination)
            method = "public_hugging_face_revision_pinned"
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"Automatic VN-MSASar v3 download failed ({type(error).__name__}). "
                f"Download ALL source split Parquet files and README.md from {source_url}/tree/{config.dataset_revision} "
                "to a local directory, then run: python -m src.cli food-data-import --dataset vn_msasar_v3 --input <directory>. "
                "No other dataset has been substituted."
            ) from None
    if not files:
        raise ValueError("Source revision contains no Parquet files; no dataset substituted")
    frames = [read_source(path) for path in files]
    for frame in frames:
        missing = set(REQUIRED) - set(frame)
        if missing:
            raise ValueError(f"VN-MSASar v3 required columns missing: {sorted(missing)}")
    actual_schema = {column: str(dtype) for column, dtype in pd.concat(frames).dtypes.items()}
    provenance = {
        "dataset_name": dataset,
        "dataset_repo": config.dataset_repo,
        "revision": revision,
        "requested_revision": config.dataset_revision,
        "source_url": source_url,
        "license": license_name,
        "license_url": LICENSE_URL,
        "attribution": "VN-MSASar v3, phamluan; retain source platform and dataset attribution",
        "import_timestamp": datetime.now(UTC).isoformat(),
        "import_method": method,
        "files": [
            {"filename": path.name, "raw_sha256": file_checksum(path), "rows": len(frame)}
            for path, frame in zip(files, frames, strict=True)
        ],
        "actual_schema": actual_schema,
        "observed_sentiment_values": sorted(
            str(value) for value in pd.concat(frames).sentiment.dropna().unique()
        ),
        "observed_aspects": sorted(pd.concat(frames).aspect.dropna().unique().tolist()),
        "label_semantics_source": SOURCE_URL,
    }
    if card_path.exists():
        provenance["dataset_card_sha256"] = file_checksum(card_path)
    atomic_json(root / "provenance.json", provenance)
    return provenance


def source_data(config):
    root = config.data_dir / "source/vn_msasar_v3"
    path = root / "provenance.json"
    if not path.exists():
        raise ValueError("Real VN-MSASar v3 source missing; run food-data-import first")
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if provenance["dataset_name"] != "vn_msasar_v3":
        raise ValueError("Primary dataset must be VN-MSASar v3")
    if provenance["dataset_repo"] != config.dataset_repo or (
        provenance.get("requested_revision", provenance["revision"]) != config.dataset_revision
        and provenance["revision"] != config.dataset_revision
    ):
        raise ValueError("Dataset location/revision changed; run food-data-import again")
    if (
        "dataset_card_sha256" in provenance
        and file_checksum(root / "README.md") != provenance["dataset_card_sha256"]
    ):
        raise ValueError("Dataset card checksum changed; explicitly re-import")
    frames = []
    for entry in provenance["files"]:
        path = root / entry["filename"]
        if file_checksum(path) != entry["raw_sha256"]:
            raise ValueError("Raw source checksum changed; explicitly re-import")
        frame = read_source(path)
        frame["original_split"] = (
            entry["filename"].split("-")[0]
            if entry["filename"].split("-")[0] in {"train", "validation", "test"}
            else None
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), provenance


def prepare(config):
    raw, provenance = source_data(config)
    rows, rejected, counts = adapt_vn_msasar(raw, provenance)
    path = config.data_dir / "processed/reviews.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(path, index=False)
    atomic_json(config.data_dir / "rejected/rows.json", rejected)
    metadata = {
        **provenance,
        **counts,
        "processed_sha256": file_checksum(path),
        "canonical_schema": {column: str(dtype) for column, dtype in rows.dtypes.items()},
    }
    atomic_json(config.data_dir / "metadata/preparation.json", metadata)
    atomic_json(
        config.reports_dir / "food_label_mapping.json",
        {
            "task": "aspect-conditioned three-class sentiment; ordinal intensity collapsed by sign",
            "mapping": LABEL_MAPPING,
            "observed_source_values": provenance["observed_sentiment_values"],
            "observed_filtered_values": sorted(rows.original_sentiment.unique().tolist()),
            "semantics": "Published signed ordinal sentiment scale -3..+3; negative scores -> negative, zero -> neutral, positive scores -> positive",
            "semantics_source": provenance["label_semantics_source"],
            "revision": provenance["revision"],
            "original_sentiment_preserved": True,
            "ratings_used": False,
        },
    )
    return metadata


def load_prepared(config):
    _, provenance = source_data(config)  # Validate raw checksums as well.
    path = config.data_dir / "processed/reviews.parquet"
    metadata_path = config.data_dir / "metadata/preparation.json"
    if not path.exists() or not metadata_path.exists():
        raise ValueError("Prepared dataset missing; run food-prepare")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["processed_sha256"] != file_checksum(path)
        or metadata["files"] != provenance["files"]
        or metadata["import_timestamp"] != provenance["import_timestamp"]
    ):
        raise ValueError("Dataset/provenance changed; rerun food-prepare and food-split")
    return pd.read_parquet(path), metadata
