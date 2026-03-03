FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Install GABRIEL from the local source tree ────────────────────────────────
# The entire repo is the build context so we get the real source, not a PyPI
# snapshot.  Any edit to prompts / tasks / utils is reflected on next build.
WORKDIR /gabriel-src
COPY . .
RUN pip install --no-cache-dir .

# ── Install the lightweight API server dependencies ───────────────────────────
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    "uvicorn[standard]==0.32.0" \
    gunicorn==23.0.0 \
    python-dotenv==1.0.1 \
    pandas==2.2.3

# ── Copy and run the OpenQDA API server ───────────────────────────────────────
WORKDIR /app
COPY server/ .

EXPOSE 80

CMD ["gunicorn", "main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:80"]
