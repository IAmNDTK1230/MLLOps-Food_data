from databricks.sdk.errors import InternalError, PermissionDenied, TooManyRequests

from src.utils.retry import transient


def test_retry_only_cloud_transient_errors():
    assert transient(InternalError("transient"))
    assert transient(TooManyRequests("transient cloud quota"))
    assert not transient(PermissionDenied("credentials"))
    assert not transient(ValueError("invalid data"))
