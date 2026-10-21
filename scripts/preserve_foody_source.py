"""Create a write-once, byte-identical Foody source snapshot and manifest."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def digest(data):
    return hashlib.sha256(data).hexdigest()


def preserve(source, snapshot, expected_sha256=None):
    source, snapshot = Path(source), Path(snapshot)
    manifest_path = snapshot.with_suffix(".manifest.json")
    if source.resolve() in {snapshot.resolve(), manifest_path.resolve()}:
        raise ValueError("Snapshot and manifest must differ from the immutable source")
    before = source.read_bytes()
    source_sha = digest(before)
    before.decode("utf-8-sig", errors="strict")
    if expected_sha256 is not None and source_sha != expected_sha256:
        raise ValueError("Source differs from the expected audited SHA256; no snapshot written")
    if snapshot.exists() or manifest_path.exists():
        if not snapshot.exists() or not manifest_path.exists():
            raise ValueError(
                "Incomplete snapshot/manifest pair; manual review required, no overwrite"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            digest(snapshot.read_bytes()) != source_sha
            or manifest["sha256"] != source_sha
            or manifest["size_bytes"] != len(before)
            or Path(manifest["source_path"]).resolve() != source.resolve()
            or Path(manifest["snapshot_path"]).resolve() != snapshot.resolve()
        ):
            raise ValueError("Existing snapshot/manifest conflicts; refusing overwrite")
        if digest(source.read_bytes()) != source_sha:
            raise ValueError("Source changed during snapshot verification")
        return manifest
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacement even if another process races us.
    with snapshot.open("xb") as stream:
        stream.write(before)
        stream.flush()
        os.fsync(stream.fileno())
    if digest(snapshot.read_bytes()) != source_sha or digest(source.read_bytes()) != source_sha:
        raise ValueError("Source/snapshot changed; no accepted manifest created")
    now = datetime.now(UTC)
    manifest = {
        "source_path": str(source.resolve()),
        "snapshot_path": str(snapshot.resolve()),
        "sha256": source_sha,
        "size_bytes": len(before),
        "snapshot_created_at_utc": now.isoformat(),
        "snapshot_created_at_asia_ho_chi_minh": now.astimezone(
            ZoneInfo("Asia/Ho_Chi_Minh")
        ).isoformat(),
        "source_modified_at_utc": datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat(),
        "source_sha256_before": source_sha,
        "source_sha256_after": digest(source.read_bytes()),
        "byte_identical": True,
        "encoding_check": "STRICT_UTF8_PASS",
        "method": "exclusive binary byte copy; no parsing, rewriting, repairs or index selection",
        "restaurant_id_provenance": "UNRESOLVED; snapshot preserves both raw headers",
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/foody/source/review.csv")
    parser.add_argument("--snapshot", default="data/foody/source/review_utf8_20261020.csv")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print(json.dumps(preserve(args.source, args.snapshot, args.expected_sha256), indent=2))


if __name__ == "__main__":
    main()
