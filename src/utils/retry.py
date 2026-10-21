from databricks.sdk.errors import (
    DeadlineExceeded,
    InternalError,
    TemporarilyUnavailable,
    TooManyRequests,
)
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def transient(error: BaseException) -> bool:
    # Authorization, invalid requests and website restrictions must never be retried.
    return isinstance(
        error,
        (
            ConnectionError,
            TimeoutError,
            RequestsConnectionError,
            RequestsTimeout,
            InternalError,
            TemporarilyUnavailable,
            DeadlineExceeded,
            TooManyRequests,
        ),
    )


retry_transient = retry(
    retry=retry_if_exception(transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    reraise=True,
)
