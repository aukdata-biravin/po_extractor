# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── System dependencies ────────────────────────────────────────────────────────
# tesseract-ocr        : OCR engine used by pytesseract
# tesseract-ocr-eng    : English language data pack
# libgl1 / libglib2.0-0 / libgomp1 : required by PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user (security hardening) ────────────────────────────────────────
RUN groupadd --gid 1001 appuser \
    && useradd  --uid 1001 --gid 1001 --no-create-home appuser

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /app

COPY requirements-po-extraction.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements-po-extraction.txt

# ── Application code ──────────────────────────────────────────────────────────
# .dockerignore prevents .venv, __pycache__, local PDFs, test data, etc.
COPY . .

# ── Output directory (owned by appuser) ───────────────────────────────────────
RUN mkdir -p /app/output && chown -R appuser:appuser /app

# USER appuser

# ── Health-check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# ── Expose & start ────────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "warning", \
     "--timeout-keep-alive", "30"]
