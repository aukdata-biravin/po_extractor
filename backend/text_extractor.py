"""
Text extraction from PDF files using PyMuPDF (fitz).
Produces clean, properly-ordered plain text — blocks are sorted top-to-bottom,
left-to-right so the AI parser sees the document in reading order.
"""

from pathlib import Path
from typing import Optional
import logging
import re

import pymupdf as fitz

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extracts text from PDF files using PyMuPDF."""

    @staticmethod
    def extract_text(pdf_path: str | Path) -> str:
        """
        Extract all text from a PDF in reading order using PyMuPDF.

        Strategy per page:
          - Use get_text("blocks") to get each text block with its bounding box.
          - Sort blocks top-to-bottom (then left-to-right) so reading order is preserved.
          - Join all pages with a double newline separator.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text as a single string
        """
        try:
            doc = fitz.open(pdf_path)
            page_texts = []
            for page in doc:
                page_text = TextExtractor._extract_page(page)
                if page_text.strip():
                    page_texts.append(page_text)
            doc.close()
            return "\n\n".join(page_texts).strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    @staticmethod
    def _extract_page(page) -> str:
        """
        Extract text from a single PyMuPDF page using a spatial character grid.

        Each word's X position is mapped to a character column so that the
        output text mirrors the visual layout of the PDF, e.g.:

            SUNRISE ENTERPRISE                         # Inv. No. : Inv-5
            Bill To                                    Inv. Date : 10-01-25
            Name: Rajiv Gupta                          Payment Mode : UPI

        This preserves side-by-side boxes, table columns, and margins using
        plain spaces — no pipes, no heuristic column detection needed.
        Vertical gaps larger than ~1.5 lines produce a blank line separator.
        """
        from collections import defaultdict

        # All words on the page: (x0, y0, x1, y1, text, ...)
        words = [w for w in page.get_text("words") if w[4].strip()]
        if not words:
            return ""

        # ── Estimate line height from median word height ──────────────────────
        heights = sorted(w[3] - w[1] for w in words)
        line_h = max(heights[len(heights) // 2], 4.0)

        # ── Character grid: derive char_w from actual text widths ─────────────
        # Using page_width / fixed_cols is inaccurate when text is dense.
        # Instead, measure the average character width from every word on the page.
        total_text_width = sum(w[2] - w[0] for w in words)
        total_chars      = sum(len(w[4]) for w in words)
        avg_char_w = total_text_width / max(total_chars, 1)
        # Clamp to a sensible range (1–8 pt per char)
        avg_char_w = max(1.0, min(avg_char_w, 8.0))

        COLS   = int(page.rect.width / avg_char_w) + 10
        char_w = page.rect.width / COLS

        # ── Group words into lines based on Y proximity ──────────────────────
        # Sort words by top Y coordinate
        sorted_words = sorted(words, key=lambda x: x[1])
        y_groups: list[list] = []
        if sorted_words:
            current_group = [sorted_words[0]]
            y_groups.append(current_group)
            
            # Grouping tolerance: 40% of median line height
            tolerance = line_h * 0.4
            
            for i in range(1, len(sorted_words)):
                w = sorted_words[i]
                prev_w = current_group[-1]
                
                # If word is within tolerance of the last word in group, add to group
                if w[1] - prev_w[1] <= tolerance:
                    current_group.append(w)
                else:
                    current_group = [w]
                    y_groups.append(current_group)

        # ── Build output lines ────────────────────────────────────────────────
        output_lines: list[str] = []
        last_y: float | None = None

        for group in y_groups:
            # Median Y of the group for gap calculation
            group_y = sorted(w[1] for w in group)[len(group)//2]

            # Blank line for vertical gaps > 1.5 lines
            if last_y is not None and group_y - last_y > line_h * 1.5:
                output_lines.append("")

            # Place each word left-to-right using a cursor so that:
            # 1. Words that map to the same column don't overwrite each other.
            # 2. The spatial gap between columns is preserved.
            line_chars = [" "] * COLS
            cursor = 0   # rightmost filled position + 1

            for w in sorted(group, key=lambda x: x[0]):
                col = int(w[0] / char_w)
                # Never go behind the cursor — prevents overwriting
                start = max(col, cursor)
                for i, ch in enumerate(w[4]):
                    pos = start + i
                    if 0 <= pos < COLS:
                        line_chars[pos] = ch
                cursor = start + len(w[4]) + 1   # +1 ensures at least one space gap

            line_str = "".join(line_chars).rstrip()
            if line_str.strip():
                output_lines.append(line_str)

            last_y = group_y

        return "\n\n".join(output_lines)

    @staticmethod
    def clean_extraction_text(text: str) -> str:
        """
        Clean extracted text while preserving spatial layout.
        Multiple spaces are kept — they carry positional meaning in the
        layout-preserving output (columns, margins, side-by-side boxes).
        Only the most obvious noise is removed: pipe chars and watermarks.
        """
        # Remove any residual pipe characters (from older extraction runs)
        text = re.sub(r'\s*\|\s*', ' ', text)
        # Collapse runs of 3+ blank lines → single blank line
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove common PDF generator watermarks
        text = re.sub(
            r'This document was created.*?(?:https?://\S+)',
            '', text, flags=re.DOTALL | re.IGNORECASE
        )
        return text.strip()

    @staticmethod
    def extract_dates_by_label(text: str) -> dict:
        """
        Extract PO Date and Expiry Date by their explicit labels.
        Works for any PO format — no vendor-specific logic.

        Supported date formats:
          - DD-Mon-YYYY  (e.g. 7-Nov-2025)
          - DD/MM/YYYY   (e.g. 07/11/2025)
          - DD-MM-YYYY   (e.g. 07-11-2025)
          - YYYY-MM-DD   (e.g. 2025-11-07)
        """
        dates = {"poDate": None, "expiryDate": None}

        # Flexible date pattern
        DATE_PATTERN = (
            r'(\d{1,2}[-/]\w+[-/]\d{2,4}'   # DD-Mon-YYYY or DD/MM/YYYY
            r'|\d{4}[-/]\d{2}[-/]\d{2})'      # YYYY-MM-DD
        )

        # PO Date labels
        po_date_match = re.search(
            rf'(?:PO\s*Date|Order\s*Date|Issued\s*On|Invoice\s*Date'
            rf'|(?<!\w)Date)\s*:?\s*{DATE_PATTERN}',
            text, re.IGNORECASE
        )
        if po_date_match:
            dates["poDate"] = po_date_match.group(1)
            logger.info(f"Extracted PO Date: {dates['poDate']}")

        # Expiry Date labels
        expiry_match = re.search(
            rf'(?:P\.O\.?\s*Expiry|Expiry\s*Date|Valid\s*Till'
            rf'|Valid\s*Until|Due\s*Date|Delivery\s*By)\s*:?\s*{DATE_PATTERN}',
            text, re.IGNORECASE
        )
        if expiry_match:
            dates["expiryDate"] = expiry_match.group(1)
            logger.info(f"Extracted Expiry Date: {dates['expiryDate']}")

        # If only one date found and two dates exist in text, try to get second
        if dates["poDate"] and not dates["expiryDate"]:
            all_dates = re.findall(DATE_PATTERN, text)
            for d in all_dates:
                if d != dates["poDate"]:
                    dates["expiryDate"] = d
                    logger.info(f"Inferred Expiry Date from remaining dates: {d}")
                    break

        return dates

    @staticmethod
    def extract_parties(text: str) -> dict:
        """
        Identify buyer and seller purely by document labels and structure.
        No vendor-specific or hardcoded business logic.

        Strategy:
          1. 'Vendor'/'Supplier' label → seller
          2. 'Bill To'/'Buyer'/'Customer' label → buyer
          3. Fallback: first company block = buyer, second = seller
        """
        parties = {
            "buyer":  {"company": None, "phone": None, "gstno": None, "tin": None, "address": None},
            "seller": {"company": None, "phone": None, "gstno": None, "tin": None, "address": None}
        }

        COMPANY_PATTERN = r'([A-Z][A-Z0-9\s\.,&\-\'\/]+?)'
        GSTIN_PATTERN   = r'[A-Z0-9]{15}'
        PHONE_PATTERN   = r'\d{10,12}'

        # ── SELLER: look for Vendor/Supplier label ──────────────────────────
        seller_label_match = re.search(
            rf'(?:Vendor|Supplier)\s+(?!Copy)(?:\n)?\s*{COMPANY_PATTERN}'
            rf'(?:\n|Phone|Address|TIN|GSTIN|$)',
            text, re.IGNORECASE | re.MULTILINE
        )
        if seller_label_match:
            parties["seller"]["company"] = seller_label_match.group(1).strip()
            logger.info(f"Seller by label: {parties['seller']['company']}")

        # ── BUYER: look for Bill To/Buyer/Customer label ────────────────────
        buyer_label_match = re.search(
            rf'(?:Bill\s*To|Buyer|Sold\s*To|Customer|Ship\s*To)\s*\n?\s*{COMPANY_PATTERN}'
            rf'(?:\n|Phone|Address|TIN|GSTIN|$)',
            text, re.IGNORECASE | re.MULTILINE
        )
        if buyer_label_match:
            parties["buyer"]["company"] = buyer_label_match.group(1).strip()
            logger.info(f"Buyer by label: {parties['buyer']['company']}")

        # ── FALLBACK: extract all company+phone blocks ───────────────────────
        if not parties["buyer"]["company"] or not parties["seller"]["company"]:
            company_blocks = re.findall(
                rf'^(?:.*?\)\s*)?{COMPANY_PATTERN}\s+Phone\s*:\s*({PHONE_PATTERN})',
                text, re.MULTILINE | re.IGNORECASE
            )

            for company, phone in company_blocks:
                company = company.strip()
                if any(skip in company.upper() for skip in [
                    'PURCHASE', 'ORDER', 'INVOICE', 'SUPPLIER', 'VENDOR'
                ]):
                    continue

                if (parties["seller"]["company"] and
                        company.upper() in parties["seller"]["company"].upper()):
                    parties["seller"]["phone"] = phone
                    continue

                if not parties["buyer"]["company"]:
                    parties["buyer"]["company"] = company
                    parties["buyer"]["phone"] = phone
                    logger.info(f"Buyer by fallback: {company}")
                elif not parties["seller"]["company"]:
                    parties["seller"]["company"] = company
                    parties["seller"]["phone"] = phone
                    logger.info(f"Seller by fallback: {company}")

        # ── GSTIN extraction (only GSTIN/GSTN labels → gstno) ──────────────────
        all_gstins = re.findall(
            rf'(?:GSTIN|GSTN)\s*:?\s*({GSTIN_PATTERN})',
            text, re.IGNORECASE
        )

        if len(all_gstins) >= 1:
            for gstin_val in all_gstins:
                pos = text.find(gstin_val)
                buyer_pos  = text.find(parties["buyer"]["company"])  if parties["buyer"]["company"]  else -1
                seller_pos = text.find(parties["seller"]["company"]) if parties["seller"]["company"] else -1

                if buyer_pos != -1 and seller_pos != -1:
                    if abs(pos - buyer_pos) < abs(pos - seller_pos):
                        if not parties["buyer"]["gstno"]:
                            parties["buyer"]["gstno"] = gstin_val
                    else:
                        if not parties["seller"]["gstno"]:
                            parties["seller"]["gstno"] = gstin_val
                elif buyer_pos != -1 and not parties["buyer"]["gstno"]:
                    parties["buyer"]["gstno"] = gstin_val
                elif seller_pos != -1 and not parties["seller"]["gstno"]:
                    parties["seller"]["gstno"] = gstin_val

        # ── TIN extraction (only TIN label → tin) ────────────────────────────
        all_tins = re.findall(
            rf'(?<![A-Z])TIN\s*:?\s*({GSTIN_PATTERN})',
            text, re.IGNORECASE
        )

        if len(all_tins) >= 1:
            for tin_val in all_tins:
                pos = text.find(tin_val)
                buyer_pos  = text.find(parties["buyer"]["company"])  if parties["buyer"]["company"]  else -1
                seller_pos = text.find(parties["seller"]["company"]) if parties["seller"]["company"] else -1

                if buyer_pos != -1 and seller_pos != -1:
                    if abs(pos - buyer_pos) < abs(pos - seller_pos):
                        if not parties["buyer"]["tin"]:
                            parties["buyer"]["tin"] = tin_val
                    else:
                        if not parties["seller"]["tin"]:
                            parties["seller"]["tin"] = tin_val
                elif buyer_pos != -1 and not parties["buyer"]["tin"]:
                    parties["buyer"]["tin"] = tin_val
                elif seller_pos != -1 and not parties["seller"]["tin"]:
                    parties["seller"]["tin"] = tin_val

        logger.info(f"Final parties — Buyer: {parties['buyer']['company']} | "
                    f"Seller: {parties['seller']['company']}")
        return parties

    @staticmethod
    def extract_po_number(text: str) -> Optional[str]:
        """
        Extract PO Number by label. Works for any PO format.
        """
        match = re.search(
            r'(?:P\.O\.?\s*(?:Number|No\.?)|PO\s*(?:Number|No\.?)'
            r'|Order\s*(?:Number|No\.?)|Ref(?:erence)?\s*(?:Number|No\.))'
            r'\s*:?\s*([\w,\-\/]+)',
            text, re.IGNORECASE
        )
        if match:
            po_number = match.group(1).strip()
            logger.info(f"Extracted PO Number: {po_number}")
            return po_number
        return None

    @staticmethod
    def build_confirmed_section(
        dates: dict,
        parties: dict,
        po_number: Optional[str]
    ) -> str:
        """
        Build the CONFIRMED VALUES section to inject into the Mistral prompt.
        Pre-extracted values take priority over AI extraction.
        """
        lines = ["## CONFIRMED PRE-EXTRACTED VALUES",
                 "These were extracted by code. Use directly. Do NOT re-extract.\n"]

        if po_number:
            lines.append(f"- poNumber: {po_number}")
        if dates.get("poDate"):
            lines.append(f"- poDate: {dates['poDate']}")
        if dates.get("expiryDate"):
            lines.append(f"- expiryDate: {dates['expiryDate']}")
        if parties["buyer"].get("company"):
            lines.append(f"- buyer.companyName: {parties['buyer']['company']}")
        if parties["buyer"].get("gstno"):
            lines.append(f"- buyer.gstno: {parties['buyer']['gstno']}")
        if parties["buyer"].get("tin"):
            lines.append(f"- buyer.tin: {parties['buyer']['tin']}")
        if parties["buyer"].get("phone"):
            lines.append(f"- buyer.phone: {parties['buyer']['phone']}")
        if parties["seller"].get("company"):
            lines.append(f"- seller.companyName: {parties['seller']['company']}")
        if parties["seller"].get("gstno"):
            lines.append(f"- seller.gstno: {parties['seller']['gstno']}")
        if parties["seller"].get("tin"):
            lines.append(f"- seller.tin: {parties['seller']['tin']}")
        if parties["seller"].get("phone"):
            lines.append(f"- seller.phone: {parties['seller']['phone']}")

        return "\n".join(lines)
