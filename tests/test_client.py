from unittest.mock import Mock

import pytest

from src.config import Settings
from src.databricks.client import DatabricksAuthenticationError, workspace_client


def test_explicit_profile_has_priority_over_legacy_pat(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("src.databricks.client.WorkspaceClient", factory)
    settings = Settings(
        databricks_config_profile="shopee",
        databricks_host="https://legacy.invalid",
        databricks_token="secret",
    )
    assert workspace_client(settings) is factory.return_value
    factory.assert_called_once_with(profile="shopee")


def test_default_unified_auth_without_pat(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("src.databricks.client.WorkspaceClient", factory)
    assert workspace_client(Settings()) is factory.return_value
    factory.assert_called_once_with()


def test_legacy_pat_only_after_default_configuration_failure(monkeypatch):
    expected = Mock()
    factory = Mock(side_effect=[ValueError("default unavailable"), expected])
    monkeypatch.setattr("src.databricks.client.WorkspaceClient", factory)
    assert (
        workspace_client(
            Settings(databricks_host="https://legacy.invalid", databricks_token="secret")
        )
        is expected
    )
    assert factory.call_args_list[0].kwargs == {}
    assert factory.call_args_list[1].kwargs == {"host": "https://legacy.invalid", "token": "secret"}


def test_invalid_explicit_profile_never_falls_back_and_hides_sdk_context(monkeypatch):
    factory = Mock(side_effect=ValueError("OAuth access_token=secret"))
    monkeypatch.setattr("src.databricks.client.WorkspaceClient", factory)
    with pytest.raises(DatabricksAuthenticationError) as error:
        workspace_client(
            Settings(
                databricks_config_profile="missing",
                databricks_host="https://legacy.invalid",
                databricks_token="secret",
            )
        )
    assert "secret" not in str(error.value)
    assert error.value.__suppress_context__
    factory.assert_called_once_with(profile="missing")


def test_no_credentials_failure_is_sanitized(monkeypatch):
    monkeypatch.setattr(
        "src.databricks.client.WorkspaceClient", Mock(side_effect=ValueError("token=secret"))
    )
    with pytest.raises(DatabricksAuthenticationError, match="unified authentication"):
        workspace_client(Settings())
