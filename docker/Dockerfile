# CAIRNS Recommendation API — image contains the agent core + FastAPI server.
# Qdrant and the LLM backend run as separate services (see docker-compose.yml / helm chart).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/cairns

WORKDIR /app

# System deps kept minimal.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY cairns/     ./cairns/
COPY app/        ./app/
COPY db_builder/ ./db_builder/
COPY scripts/    ./scripts/
# Strip CR: the repo is edited on Windows and CRLF breaks bash inside the image.
RUN sed -i 's/\r$//' scripts/*.sh && chmod +x scripts/*.sh

# Non-root. /data holds the KG sqlite + build artifacts (mounted volume/PVC in prod).
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin cairns && \
    mkdir -p /data && chown -R cairns:cairns /app /data
USER 10001

EXPOSE 8000

# ROOT_PATH mounts the API under a subpath (e.g. /cairns) behind a gateway; empty = root.
ENV ROOT_PATH=""

# Serve the API. Index build is a separate one-off step (see README / compose / helm Job).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]
