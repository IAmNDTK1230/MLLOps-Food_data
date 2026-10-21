from src.storage.manifest import UploadManifest


def test_destination_and_checksum_both_required(tmp_path):
    manifest = UploadManifest(tmp_path / "manifest.json")
    manifest.add({"remote_path": "/v/a", "checksum": "123", "status": "UPLOADED"})
    assert manifest.contains("/v/a", "123")
    assert not manifest.contains("/v/b", "123")
    assert not manifest.contains("/v/a", "changed")
    manifest.add({"remote_path": "/v/c", "checksum": "123", "status": "FAILED"})
    assert not manifest.contains("/v/c", "123")
