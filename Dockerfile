# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

# System deps kept minimal on purpose — every extra apt package costs disk
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
