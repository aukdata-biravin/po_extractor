"""
Main PO Extraction Orchestrator.
Coordinates the complete workflow: OCR detection, text extraction, and AI parsing.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from config import settings
from ocr_handler import OCRHandler
from text_extractor import TextExtractor
from ai_parser import AIParser
from schemas.purchase_order import PurchaseOrder, PurchaseOrderResponse

logger = logging.getLogger(__name__)


class POExtractor:
    """
    Main orchestrator for Purchase Order extraction.

    Thread-safety note: instances are stateless between calls — a single
    shared instance is safe to use from multiple async workers.
    """

    def __init__(self) -> None:
        """Initialise sub-components from the global settings singleton."""
        settings.validate()
        self.ocr_handler = OCRHandler()
        self.text_extractor = TextExtractor()
        self.ai_parser = AIParser()
        logger.info("POExtractor initialised (model=%s)", settings.model_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        save_extracted_text: bool = True,
        output_text_path: Optional[str | Path] = None,
        output_json_path: Optional[str | Path] = None,
        output_ocr_pdf_path: Optional[str | Path] = None,
    ) -> PurchaseOrderResponse:
        """
        Full extraction workflow for a single PDF.

        Steps
        -----
        1. Validate the file exists.
        2. Detect whether the PDF is searchable or scanned.
        3. If scanned, apply Tesseract OCR to produce a searchable PDF.
        4. Extract text with PyMuPDF spatial-grid extractor.
        5. Run label-based pre-extraction (dates, parties, PO number).
        6. Optionally save the cleaned text.
        7. Send text to AI parser (two-pass: header + line items).
        8. Validate line items with business rules.
        9. Optionally save the JSON output.
        10. Return a :class:`PurchaseOrderResponse`.

        Parameters
        ----------
        pdf_path:
            Path to the source PDF.
        save_extracted_text:
            If *True*, write the cleaned text to *output_text_path*.
        output_text_path:
            Destination for the cleaned text file.
            Defaults to ``{stem}_extracted.txt`` beside the PDF.
        output_json_path:
            Destination for the JSON result.
            Defaults to ``{stem}_po.json`` beside the PDF.
        output_ocr_pdf_path:
            Destination for the searchable PDF created by OCR.
            If *None* for a scanned PDF, a temporary file is created and
            deleted after extraction.

        Returns
        -------
        PurchaseOrderResponse
            Validated, structured Purchase Order.

        Raises
        ------
        FileNotFoundError
            If the PDF does not exist.
        ValueError
            If text extraction or AI parsing fails.
        """
        pdf_path = Path(pdf_path).resolve()

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        ocr_output_path: Optional[Path] = None
        should_cleanup_ocr = False

        try:
            # Step 2: Searchability check
            is_searchable = self.ocr_handler.is_searchable_pdf(pdf_path)
            logger.info("PDF is %s: %s", "searchable" if is_searchable else "scanned", pdf_path.name)

            # Step 3: OCR when needed
            pdf_to_extract = pdf_path
            if not is_searchable:
                if output_ocr_pdf_path:
                    ocr_output_path = Path(output_ocr_pdf_path)
                    should_cleanup_ocr = False
                else:
                    ocr_output_path = pdf_path.parent / f"{pdf_path.stem}_ocr_tmp.pdf"
                    should_cleanup_ocr = True

                try:
                    pdf_to_extract = self.ocr_handler.apply_ocr(
                        pdf_path, output_path=ocr_output_path
                    )
                    logger.info("OCR complete → %s", pdf_to_extract.name)
                except RuntimeError as ocr_err:
                    logger.warning(
                        "OCR failed (%s). Attempting extraction on original PDF.", ocr_err
                    )
                    pdf_to_extract = pdf_path
                    should_cleanup_ocr = False

            # Step 4: Text extraction
            extracted_text = self.text_extractor.extract_text(pdf_to_extract)
            logger.info("Extracted %d characters of text", len(extracted_text))

            if not extracted_text.strip():
                raise ValueError("No text could be extracted from the PDF.")

            # Step 5: Label-based pre-extraction
            dates = self.text_extractor.extract_dates_by_label(extracted_text)
            parties = self.text_extractor.extract_parties(extracted_text)
            po_number = self.text_extractor.extract_po_number(extracted_text)
            logger.info(
                "Pre-extracted — PO#: %s | PO Date: %s | Buyer: %s | Seller: %s",
                po_number,
                dates.get("poDate"),
                parties["buyer"].get("company"),
                parties["seller"].get("company"),
            )

            cleaned_text = self.text_extractor.clean_extraction_text(extracted_text)
            confirmed_section = self.text_extractor.build_confirmed_section(
                dates=dates, parties=parties, po_number=po_number
            )

            # Step 6: Optionally persist the cleaned text
            if save_extracted_text:
                text_dest = (
                    Path(output_text_path)
                    if output_text_path
                    else pdf_path.parent / f"{pdf_path.stem}_extracted.txt"
                )
                self._safe_write_text(text_dest, cleaned_text, label="extracted text")

            # Step 7: AI parsing
            purchase_order: PurchaseOrder = self.ai_parser.parse_po_text(
                cleaned_text, confirmed_section=confirmed_section
            )
            logger.info("AI parsing succeeded")

            # Step 8: Business-rule validation
            val = self.ai_parser.validate_line_items(purchase_order)
            if val["errors"]:
                for err in val["errors"]:
                    logger.error("Validation error: %s", err)
            if val["warnings"]:
                for warn in val["warnings"]:
                    logger.warning("Validation warning: %s", warn)
            logger.info("Validated %d line item(s)", val["itemCount"])

            # Step 9: Persist JSON
            json_dest = (
                Path(output_json_path)
                if output_json_path
                else pdf_path.parent / f"{pdf_path.stem}_po.json"
            )
            response = PurchaseOrderResponse(purchaseOrder=purchase_order)
            self._safe_write_text(
                json_dest,
                json.dumps(response.model_dump(), indent=2),
                label="JSON output",
            )

            return response

        except Exception:
            logger.exception("Error extracting PO from %s", pdf_path.name)
            raise
        finally:
            if should_cleanup_ocr and ocr_output_path and ocr_output_path.exists():
                try:
                    ocr_output_path.unlink()
                    logger.debug("Removed temporary OCR file: %s", ocr_output_path)
                except Exception as cleanup_err:
                    logger.warning("Could not delete temp OCR file: %s", cleanup_err)

    def get_pdf_info(self, pdf_path: str | Path) -> dict:
        """
        Return metadata and searchability status for a PDF.

        Raises
        ------
        FileNotFoundError
            If *pdf_path* does not exist.
        """
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        metadata = self.ocr_handler.get_pdf_metadata(pdf_path)
        metadata["is_searchable"] = self.ocr_handler.is_searchable_pdf(pdf_path)
        return metadata

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _safe_write_text(dest: Path, content: str, label: str = "file") -> None:
        """Write *content* to *dest*, creating parent dirs as needed.  Logs but never raises."""
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            logger.info("Saved %s to: %s", label, dest)
        except Exception as exc:
            logger.warning("Could not save %s to %s: %s", label, dest, exc)
