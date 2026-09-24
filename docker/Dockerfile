# CAIRNS Recommendation API — image contains the agent core + FastAPI server.
# Qdrant and Ollama run as separate services (see docker-compose.yml).
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
RUN chmod +x scripts/*.sh

EXPOSE 8000

# Serve the API. Index build is a separate one-off step (see README / compose).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]
