"""
Centralised application configuration.

All settings are read from environment variables (with sane defaults).
Import `settings` from this module wherever configuration is needed —
never read os.environ directly in application code.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppSettings:
    # ── API / Auth ────────────────────────────────────────────────────────────
    nvidia_api_key: str = field(
        default_factory=lambda: (
            os.getenv("NVIDIA_API_KEY") or os.getenv("nvidia_api_key") or ""
        )
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_NAME", "qwen/qwen3.5-122b-a10b"
        )
    )
    api_base_url: str = field(
        default_factory=lambda: os.getenv(
            "API_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
    )

    # ── Extraction ────────────────────────────────────────────────────────────
    ocr_dpi: int = field(
        default_factory=lambda: int(os.getenv("OCR_DPI", "400"))
    )
    ocr_language: str = field(
        default_factory=lambda: os.getenv("OCR_LANGUAGE", "en")
    )
    # Number of pages checked to decide if a PDF is searchable
    searchable_sample_pages: int = field(
        default_factory=lambda: int(os.getenv("SEARCHABLE_SAMPLE_PAGES", "3"))
    )

    # ── API tokens ────────────────────────────────────────────────────────────
    header_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("HEADER_MAX_TOKENS", "4096"))
    )
    items_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("ITEMS_MAX_TOKENS", "32000"))
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "/app/output"))
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins, e.g. "https://example.com,https://app.example.com"
    # Default "*" is intentionally permissive for internal/dev use; restrict in prod.
    cors_origins: list = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "*").split(",")
            if o.strip()
        ]
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )

    # ── Upload limits ─────────────────────────────────────────────────────────
    # Maximum accepted file size in bytes (default 50 MB)
    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
    )

    def validate(self) -> None:
        """Raise ValueError if critical settings are missing or invalid."""
        if not self.nvidia_api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Export it as an environment variable or add it to .env."
            )
        if self.ocr_dpi < 72 or self.ocr_dpi > 1200:
            raise ValueError(f"OCR_DPI must be between 72 and 1200, got {self.ocr_dpi}")
        if self.max_upload_bytes < 1:
            raise ValueError("MAX_UPLOAD_BYTES must be positive")

    def is_api_configured(self) -> bool:
        return bool(self.nvidia_api_key)


# Singleton — imported everywhere
settings = AppSettings()
