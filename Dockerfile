FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt && python -m playwright install --with-deps chromium
COPY src ./src
COPY databricks ./databricks
COPY scripts ./scripts
COPY prefect.yaml ./
RUN useradd --create-home --uid 10001 pipeline && mkdir -p data state logs && chown -R pipeline:pipeline /app /ms-playwright
USER pipeline
CMD ["python", "-m", "src.cli", "--help"]
