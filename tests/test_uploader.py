from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from databricks.sdk.errors import NotFound

from src.config import Settings
from src.databricks.uploader import Uploader


def test_upload_once_and_recover_missing_manifest(tmp_path):
    path = tmp_path / "data/raw/products/crawl_date=2026-10-03/a.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"product_id":"1"}\n')
    files = Mock()
    files.get_metadata.side_effect = NotFound("absent")
    settings = Settings(data_dir=tmp_path / "data", state_dir=tmp_path / "state")
    uploader = Uploader(SimpleNamespace(files=files), settings)
    assert uploader.upload(path, "run")["status"] == "UPLOADED"
    assert uploader.upload(path, "run")["status"] == "SKIPPED"
    assert files.upload.call_count == 1
    files.get_metadata.side_effect = None
    files.download.return_value = SimpleNamespace(contents=BytesIO(path.read_bytes()))
    uploader.manifest.path.unlink()
    assert uploader.upload(path, "run")["status"] == "UPLOADED"
    assert files.upload.call_count == 1


def test_remote_collision_never_overwrites(tmp_path):
    path = tmp_path / "data/staged/products/a.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"new")
    files = Mock()
    files.download.return_value = SimpleNamespace(contents=BytesIO(b"old"))
    uploader = Uploader(
        SimpleNamespace(files=files),
        Settings(data_dir=tmp_path / "data", state_dir=tmp_path / "state"),
    )
    with pytest.raises(RuntimeError, match="collision"):
        uploader.upload(path, "run")
    files.upload.assert_not_called()
    assert uploader.manifest.read()[0]["status"] == "FAILED"
