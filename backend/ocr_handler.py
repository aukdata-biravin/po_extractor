"""
OCR detection and handling for PDFs.
Detects if a PDF is scanned or already searchable.
If scanned, uses PyMuPDF to render pages as high-res images and Tesseract
(via pytesseract) to extract text, then assembles a searchable PDF.
No PaddlePaddle dependency — Tesseract is installed as a system package.
"""

import tempfile
from pathlib import Path
from typing import Optional
import logging

import pymupdf as fitz
# PIL and pytesseract are only installed inside Docker (via apt + pip).
# Importing them lazily inside apply_ocr() avoids IDE errors in the local venv.

logger = logging.getLogger(__name__)


class OCRHandler:
    """Handles OCR detection and processing for PDF files."""

    @staticmethod
    def is_searchable_pdf(pdf_path: str | Path, sample_pages: int = 3) -> bool:
        """
        Check if a PDF is searchable (has embedded text).

        Args:
            pdf_path: Path to the PDF file
            sample_pages: Number of pages to sample for text detection

        Returns:
            True if PDF has searchable text, False if it's scanned
        """
        logger.info(f"Checking if PDF is searchable: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            pages_to_check = min(sample_pages, len(doc))

            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text()
                # If we find substantial text, it's searchable
                if text.strip() and len(text.strip()) > 50:
                    doc.close()
                    logger.info("PDF is searchable")
                    return True

            doc.close()
            logger.info("PDF is scanned")
            return False

        except Exception as e:
            logger.error(f"Error checking if PDF is searchable: {e}")
            return False

    @staticmethod
    def apply_ocr(
        pdf_path: str | Path,
        output_path: Optional[str | Path] = None,
        language: str = "en",
        dpi: int = 400,
    ) -> Path:
        """
        Apply OCR to a scanned PDF using PyMuPDF + Tesseract.

        Pipeline:
          1. Open the PDF with PyMuPDF.
          2. Render each page to a high-resolution PNG image.
          3. Run Tesseract on the image to extract plain text.
          4. Build a new searchable PDF where each page reproduces the original
             image as background and overlays the recognised text as invisible
             (white, size-1) text so downstream text extraction works.

        Args:
            pdf_path: Path to the scanned PDF.
            output_path: Where to write the searchable PDF.
                         Defaults to {stem}_searchable.pdf alongside the input.
            language: Tesseract language code (e.g. "eng", "hin").
                      Short codes "en"/"hi" are mapped automatically.
            dpi: Resolution for rendering PDF pages to images (higher = better OCR).

        Returns:
            Path to the searchable output PDF.

        Raises:
            RuntimeError: If Tesseract is not installed or OCR fails.
        """
        pdf_path = Path(pdf_path).absolute()

        if output_path is None:
            output_path = pdf_path.parent / f"{pdf_path.stem}_searchable{pdf_path.suffix}"
        else:
            output_path = Path(output_path).absolute()

        # Map short language codes to Tesseract language codes
        _lang_map = {"en": "eng", "eng": "eng", "hi": "hin", "hin": "hin"}
        tess_lang = _lang_map.get(language, language)

        logger.info(
            f"Applying OCR (PyMuPDF + Tesseract) to: {pdf_path.name} "
            f"→ {output_path.name} (lang={tess_lang}, dpi={dpi})"
        )

        doc = fitz.open(pdf_path)
        page_count = len(doc)
        logger.info(f"PDF has {page_count} page(s) to OCR")

        out_doc = fitz.open()  # new, empty PDF for searchable output

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for page_num in range(page_count):
                page = doc[page_num]

                # Step 1: Render page to a high-res PNG image
                zoom = dpi / 72  # 72 is PyMuPDF's base DPI
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)

                img_path = tmp_path / f"page_{page_num:04d}.png"
                pix.save(str(img_path))
                logger.info(
                    f"Rendered page {page_num + 1}/{page_count} → {img_path.name}"
                )

                # Step 2: Run Tesseract — get per-word bounding boxes
                # Lazy imports — only available inside Docker
                # pyrefly: ignore [missing-import]
                from PIL import Image, ImageEnhance  # noqa: PLC0415
                # pyrefly: ignore [missing-import]
                import pytesseract    # noqa: PLC0415
                from collections import defaultdict  # noqa: PLC0415

                pil_img = Image.open(img_path)
                # Enhance image for better OCR accuracy across the entire document
                pil_img = pil_img.convert('L')  # Grayscale
                from PIL import ImageEnhance, ImageOps  # noqa: PLC0415
                # Maximize contrast range before binarization
                pil_img = ImageOps.autocontrast(pil_img)
                # Sharpen first to help with faint text in shaded regions
                pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
                # Binarize the entire page to handle shaded backgrounds, headers, and colored sections
                # Using 128 (middle) is generally safer than 150 for preserving lighter characters
                pil_img = pil_img.point(lambda p: 255 if p > 128 else 0, mode='1')

                # image_to_data gives per-word layout: left, top, width, height (pixels)
                data = pytesseract.image_to_data(
                    pil_img,
                    lang=tess_lang,
                    config="--psm 11 -c preserve_interword_spaces=1",
                    output_type=pytesseract.Output.DICT,
                )

                # Group words into lines by (block_num, par_num, line_num)
                line_groups: dict = defaultdict(list)
                for idx in range(len(data["text"])):
                    word = data["text"][idx].strip()
                    # Accept all detected text regardless of confidence score
                    if word:
                        key = (data["block_num"][idx], data["par_num"][idx], data["line_num"][idx])
                        line_groups[key].append(idx)

                logger.info(
                    f"Tesseract extracted {len(line_groups)} line(s) on page {page_num + 1}"
                )

                # Step 3: Build a searchable output page
                page_rect = page.rect
                out_page = out_doc.new_page(
                    width=page_rect.width, height=page_rect.height
                )

                # Insert the original rasterised image as the visual background
                out_page.insert_image(
                    fitz.Rect(0, 0, page_rect.width, page_rect.height),
                    filename=str(img_path),
                )

                # Scale factors: pixel → PDF points
                scale_x = page_rect.width  / pix.width
                scale_y = page_rect.height / pix.height

                # Step 4: Insert each line at its correct spatial position (invisible).
                # Using per-line coords means PyMuPDF's get_text("blocks") returns
                # properly positioned blocks → column detection & separators work.
                for key in sorted(line_groups.keys()):
                    indices = line_groups[key]
                    line_text = " ".join(
                        data["text"][i] for i in indices if data["text"][i].strip()
                    )
                    if not line_text.strip():
                        continue

                    i0 = indices[0]
                    x_pt = data["left"][i0] * scale_x
                    line_h_px = max(data["height"][i] for i in indices)
                    y_pt = (data["top"][i0] + line_h_px) * scale_y  # baseline

                    # Font size proportional to detected text height in the image
                    font_size = max(line_h_px * scale_y * 0.85, 4.0)

                    try:
                        out_page.insert_text(
                            fitz.Point(x_pt, y_pt),
                            line_text,
                            fontsize=font_size,
                            color=(1, 1, 1),   # white → invisible over image bg
                            overlay=False,
                        )
                    except Exception as ln_err:
                        logger.debug(f"Skipping line '{line_text[:30]}': {ln_err}")

        doc.close()

        out_doc.save(str(output_path))
        out_doc.close()

        logger.info(f"OCR complete. Searchable PDF saved to: {output_path}")
        return output_path

    @staticmethod
    def get_pdf_page_count(pdf_path: str | Path) -> int:
        """
        Get the total number of pages in a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Number of pages
        """
        try:
            doc = fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.error(f"Error getting page count: {e}")
            return 0

    @staticmethod
    def get_pdf_metadata(pdf_path: str | Path) -> dict:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary containing metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creationDate": doc.metadata.get("creationDate", ""),
                "modificationDate": doc.metadata.get("modificationDate", ""),
                "pages": len(doc),
            }
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
