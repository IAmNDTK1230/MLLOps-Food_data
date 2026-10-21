import os

import pytest
from pydantic import ValidationError

from src.config import Settings


def test_normal_default_catalog():
    assert Settings().databricks_catalog == "shopee"


def test_token_excluded_from_serialization():
    settings = Settings(databricks_token="secret-test-only")
    assert "secret-test-only" not in repr(settings)
    assert "databricks_token" not in settings.model_dump()


def test_profile_loaded_from_environment_without_pat(monkeypatch):
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", " shopee ")
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    settings = Settings.from_env()
    assert settings.databricks_config_profile == "shopee"
    assert settings.databricks_token == ""


@pytest.mark.parametrize("value", ["shopee; DROP TABLE a", "../../other", "a.b"])
def test_unsafe_names_rejected(value):
    with pytest.raises(ValidationError):
        Settings(databricks_catalog=value)


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_prefect_api_url_is_unset(monkeypatch, value):
    monkeypatch.setenv("PREFECT_API_URL", value)
    monkeypatch.setattr("src.config.load_dotenv", lambda: None)
    Settings.from_env()
    assert "PREFECT_API_URL" not in os.environ
