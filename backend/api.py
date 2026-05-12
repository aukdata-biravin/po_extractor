"""
FastAPI application for Purchase Order Extraction.
Provides REST API endpoints for PDF upload and PO data extraction.
"""

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# ── Load .env before importing config ─────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

from config import settings  # noqa: E402  (must come after load_dotenv)
from extractor import POExtractor  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
# Silence noisy third-party libraries
for _lib in ("uvicorn.access", "uvicorn.error", "httpx", "httpcore", "openai", "multipart"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── Extractor (module-level singleton, initialised in lifespan) ───────────────
_extractor: POExtractor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle handler."""
    global _extractor
    logger.info("🚀 Starting PO Extraction API")
    if settings.is_api_configured():
        try:
            _extractor = POExtractor()
            logger.info("✅ POExtractor initialised")
        except Exception as exc:
            logger.error("❌ Failed to initialise POExtractor: %s", exc)
    else:
        logger.warning("⚠️  NVIDIA_API_KEY not set — extraction endpoints unavailable")

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("📁 Output directory: %s", settings.output_dir)
    logger.info("🚀 API is available at: http://localhost:8000/home")
    yield
    logger.info("🛑 Shutting down PO Extraction API")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Purchase Order Extraction API",
    description="Extract structured Purchase Order data from PDF documents.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request trace ID ──────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Helper: require extractor or raise ───────────────────────────────────────
def _get_extractor() -> POExtractor:
    if _extractor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_unavailable",
                "message": "PO Extractor is not initialised. Check NVIDIA_API_KEY configuration.",
            },
        )
    return _extractor


# ── Static frontend ───────────────────────────────────────────────────────────
_frontend_dir = Path(__file__).parent / "frontend"
if _frontend_dir.exists():
    app.mount("/home", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Operations"])
async def health_check():
    """Liveness / readiness probe."""
    return {
        "status": "healthy",
        "service": "PO Extraction API",
        "version": app.version,
        "extraction_ready": _extractor is not None,
        "api_key_configured": settings.is_api_configured(),
        "model": settings.model_name,
    }


@app.post("/extract", tags=["Extraction"])
async def extract_po(
    request: Request,
    file: UploadFile = File(..., description="PDF file containing a Purchase Order"),
    save_files: bool = True,
):
    """
    Extract structured Purchase Order data from an uploaded PDF.

    - Detects whether the PDF is searchable or scanned
    - Applies Tesseract OCR if needed
    - Extracts text with PyMuPDF spatial grid
    - Parses with NVIDIA Mistral (two-pass: header then line items)

    Returns the structured JSON Purchase Order plus optional paths to
    the saved `.txt` / `.json` / searchable-PDF artefacts.
    """
    extractor = _get_extractor()
    request_id = getattr(request.state, "request_id", "-")

    # ── Validate file type ────────────────────────────────────────────────────
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # ── Read & size-check payload ─────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > settings.max_upload_bytes:
        max_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {max_mb} MB size limit.",
        )

    logger.info("[%s] Received file: %s (%d bytes)", request_id, filename, len(contents))

    # ── Process via temp file ─────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        is_searchable = extractor.ocr_handler.is_searchable_pdf(tmp_path)
        pdf_type = "searchable" if is_searchable else "scanned"
        logger.info("[%s] PDF type: %s", request_id, pdf_type)

        base_stem = Path(filename).stem
        out_text = settings.output_dir / f"{base_stem}_extracted.txt" if save_files else None
        out_json = settings.output_dir / f"{base_stem}_po.json" if save_files else None
        out_ocr = (
            settings.output_dir / f"{base_stem}_searchable.pdf"
            if (save_files and not is_searchable)
            else None
        )

        logger.info("[%s] Running extraction workflow…", request_id)
        result = extractor.extract_from_pdf(
            pdf_path=tmp_path,
            save_extracted_text=save_files,
            output_text_path=out_text,
            output_json_path=out_json,
            output_ocr_pdf_path=out_ocr,
        )
        logger.info("[%s] Extraction complete", request_id)

        response_data: dict = {
            "status": "success",
            "filename": filename,
            "pdf_type": pdf_type,
            "purchaseOrder": result.purchaseOrder.model_dump() if result.purchaseOrder else None,
        }

        return JSONResponse(response_data)

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("[%s] Validation error: %s", request_id, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("[%s] Unexpected error processing %s", request_id, filename)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as cleanup_err:
            logger.warning("[%s] Temp file cleanup failed: %s", request_id, cleanup_err)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the frontend UI."""
    return RedirectResponse(url="/home")


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=settings.log_level.lower(),
    )
