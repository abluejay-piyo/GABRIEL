FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /gabriel-src
COPY . .

# Install library and API server runtime dependencies.
RUN pip install --no-cache-dir . \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.32.0" \
    gunicorn==23.0.0 \
    python-dotenv==1.0.1

WORKDIR /app
COPY server/ .

EXPOSE 80

CMD ["gunicorn", "main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:80"]
