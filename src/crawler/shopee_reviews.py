from src.crawler.base import BaseCrawler, CrawlResult


class ShopeeReviewCrawler(BaseCrawler):
    """Extension point: public review access is not verified, so no requests are issued."""

    async def crawl(self, keyword: str, pages: int, *, resume: bool = False) -> CrawlResult:
        return CrawlResult(
            status="FAILED",
            errors=[
                "Review crawling is not implemented for the current public DOM. Disable REVIEWS_ENABLED."
            ],
        )
