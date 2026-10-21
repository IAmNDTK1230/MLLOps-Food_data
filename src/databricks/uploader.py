from pathlib import Path, PurePosixPath
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceAlreadyExists

from src.config import Settings
from src.storage.manifest import UploadManifest
from src.utils.hashing import file_checksum
from src.utils.logging import event
from src.utils.retry import retry_transient
from src.utils.timestamps import utc_now


class Uploader:
    def __init__(self, client: WorkspaceClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.manifest = UploadManifest(settings.state_dir / "upload_manifest.json")

    def remote_path(self, local: Path) -> str:
        relative = local.resolve().relative_to(self.settings.data_dir.resolve())
        if (
            len(relative.parts) < 3
            or relative.parts[0] not in {"raw", "staged"}
            or relative.parts[1]
            not in {
                "products",
                "reviews",
            }
        ):
            raise ValueError("Only raw/staged product/review files may be uploaded")
        return self.settings.volume_root + "/" + PurePosixPath(*relative.parts[1:]).as_posix()

    @retry_transient
    def _upload(self, local: Path, remote: str, checksum: str) -> None:
        self.client.files.create_directory(str(PurePosixPath(remote).parent))
        try:
            self.client.files.get_metadata(remote)
        except NotFound:
            try:
                with local.open("rb") as stream:
                    self.client.files.upload(remote, stream, overwrite=False)
                return
            except ResourceAlreadyExists:
                pass
        # Recover safely if remote upload succeeded before local manifest persistence.
        import hashlib

        response = self.client.files.download(remote)
        if response.contents is None:
            raise RuntimeError("Remote file cannot be verified")
        with response.contents as stream:
            remote_checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if remote_checksum != checksum:
            raise RuntimeError(f"Remote collision; refusing overwrite: {remote}")

    def upload(self, local: Path, run_id: str) -> dict[str, Any]:
        remote = self.remote_path(local)
        checksum = file_checksum(local)
        with self.manifest.lock:
            if self.manifest.contains(remote, checksum):
                return {"local_path": str(local), "remote_path": remote, "status": "SKIPPED"}
            record = {
                "local_path": str(local.resolve()),
                "remote_path": remote,
                "checksum": checksum,
                "uploaded_at": utc_now().isoformat(),
                "run_id": run_id,
                "status": "UPLOADED",
            }
            try:
                self._upload(local, remote, checksum)
            except Exception:
                self.manifest.add({**record, "status": "FAILED"})
                raise
            self.manifest.add(record)
        event("upload_file", run_id=run_id, remote_path=remote, status="UPLOADED")
        return record
