from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CrawlResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    status: str = "SUCCESS"
    errors: list[str] = field(default_factory=list)


class CrawlStopped(Exception):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


class TransientCrawlError(RuntimeError):
    """Temporary public-site service failure, distinct from an access restriction."""


class BaseCrawler(ABC):
    @abstractmethod
    async def crawl(self, keyword: str, pages: int, *, resume: bool = False) -> CrawlResult:
        """Return public observations or explicit restriction/failure status."""
