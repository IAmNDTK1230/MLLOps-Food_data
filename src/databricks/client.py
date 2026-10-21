from databricks.sdk import WorkspaceClient

from src.config import Settings
from src.utils.retry import retry_transient


class DatabricksAuthenticationError(RuntimeError):
    """Safe configuration failure: SDK exception text can contain credentials."""


def workspace_client(settings: Settings) -> WorkspaceClient:
    try:
        if settings.databricks_config_profile:
            # An explicitly selected profile must not silently switch identities.
            return WorkspaceClient(profile=settings.databricks_config_profile)
        try:
            return WorkspaceClient()
        except ValueError:
            # Unified auth already supports PAT environment variables. This also
            # supports legacy host/token supplied directly through Settings.
            if not settings.databricks_host or not settings.databricks_token:
                raise
            return WorkspaceClient(host=settings.databricks_host, token=settings.databricks_token)
    except Exception:
        raise DatabricksAuthenticationError(
            "Databricks authentication could not be configured; check the selected "
            "CLI profile or SDK unified authentication configuration."
        ) from None


@retry_transient
def check_connection(client: WorkspaceClient) -> None:
    client.current_user.me()
