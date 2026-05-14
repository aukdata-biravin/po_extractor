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
    def trim_for_ai(text: str) -> str:
        """
        Remove irrelevant boilerplate sections before sending text to the AI.

        Many PO documents contain large legal / T&C blocks (Terms and Conditions,
        Annexures, Vendor Code of Conduct, etc.) that follow the actual purchase-order
        data.  Sending these to the model:
          • wastes tokens (triggers 504 timeouts on large docs)
          • confuses the model (legal prose contains numbers that look like data)

        Strategy
        --------
        1. Detect the first occurrence of well-known section-boundary markers.
        2. Keep everything BEFORE that marker (the real PO data).
        3. Also capture any "Amendment Lines" / "Annexure" table that may appear
           AFTER the boilerplate, since it can contain additional line-item data.
        4. ALSO: detect mid-document legal prose between the header and items table
           (common in Reliance-format POs) and stitch header directly to items table.
        5. Return the combined trimmed text.

        The heuristic is intentionally conservative — if no marker is found the
        full text is returned unchanged.
        """
        # ── Boilerplate section-start markers (case-insensitive) ─────────────
        # These signal the start of T&C / legal prose that is NOT PO data.
        BOILERPLATE_MARKERS = [
            # ── Appear just after the totals row — cut here first ─────────────
            r'Terms\s+of\s+payment\s*:',
            r'Note\s*\(S\)\s*:',
            r'Note\s*\(s\)\s*:',
            r'Supplier\s+Note\s*:',
            r'Important\s+Note\s*:',
            # ── Full legal / T&C section headers ──────────────────────────────
            r'Terms\s+[Aa]nd\s+[Cc]onditions',
            r'TERMS\s*&\s*CONDITIONS',
            r'Vendor\s+Code\s+of\s+Conduct',
            r'Annexure[-\s]*I\b',
            r'Anti[-\s]*Bribery',
            r'Compliance\s+with\s+Laws',
            r'Environmental\s+[&and]+\s+[Ss]ocial',
            r'Business\s+Continuity',
            r'Right\s+to\s+Terminate',
            r'DEFINITIONS\s+AND\s+INTERPRETATION',
            r'GENERAL\s+CONDITIONS\s+OF\s+PURCHASE',
        ]

        # ── Amendment/Appendix section that may follow T&C ───────────────────
        # These sections can contain real line-item data and should be preserved.
        TAIL_SECTIONS = [
            r'Amendment\s+Lines?',
            r'Appendix\s+[:\-]?\s*[Ii]tems?',
            r'Schedule\s+[:\-]?\s*[Ii]tems?',
        ]

        # ── Items-table column-header anchors ────────────────────────────────
        # When found, everything from this point forward is real PO data.
        ITEMS_TABLE_ANCHORS = [
            r'Sr\.?\s*No\.?\s+Article',
            r'Sr\.?\s*No\.?\s+Item',
            r'S\.?\s*No\.?\s+Article',
            r'Sl\.?\s*No\.?\s+',
            r'Sr\.\s*No\.\s+HSN',
            r'Item\s+No\.?\s+Description',
            r'Article\s+No\.\s+EAN',
        ]

        # Find the earliest boilerplate marker position
        cutoff_pos = len(text)
        for marker in BOILERPLATE_MARKERS:
            m = re.search(marker, text, re.IGNORECASE)
            if m and m.start() < cutoff_pos:
                cutoff_pos = m.start()

        if cutoff_pos == len(text):
            # No boilerplate found — return as-is
            logger.debug("trim_for_ai: no boilerplate detected, returning full text (%d chars)", len(text))
            return text

        head = text[:cutoff_pos].strip()
        tail_text = text[cutoff_pos:]

        # Try to capture a tail section (Amendment Lines etc.) from after boilerplate
        tail_match = None
        for pattern in TAIL_SECTIONS:
            tail_match = re.search(pattern, tail_text, re.IGNORECASE)
            if tail_match:
                break

        if tail_match:
            tail = tail_text[tail_match.start():].strip()
            trimmed = head + "\n\n" + tail
        else:
            trimmed = head

        # ── PASS 2: Remove mid-document legal prose ───────────────────────────
        # Some POs (e.g. Reliance) embed a legal preamble BETWEEN the header
        # and the items table. Detect the items table header anywhere in the
        # trimmed text and, if there is a large gap of prose before it,
        # keep only: first MAX_HEADER_CHARS of the head + items table onward.
        MAX_HEADER_CHARS = 3500  # anything beyond this before the items table is prose

        items_start = None
        for anchor in ITEMS_TABLE_ANCHORS:
            m = re.search(anchor, trimmed, re.IGNORECASE)
            if m:
                if items_start is None or m.start() < items_start:
                    items_start = m.start()

        if items_start is not None and items_start > MAX_HEADER_CHARS:
            # There is a large block of prose before the items table
            # Keep the first MAX_HEADER_CHARS (header block) + items table onward
            trimmed = trimmed[:MAX_HEADER_CHARS].strip() + "\n\n" + trimmed[items_start:].strip()
            logger.info(
                "trim_for_ai: removed mid-doc prose; items table starts at char %d", items_start
            )

        removed = len(text) - len(trimmed)
        logger.info(
            "trim_for_ai: reduced text from %d → %d chars (removed %d chars / %.0f%%)",
            len(text), len(trimmed), removed, 100 * removed / max(len(text), 1)
        )
        return trimmed


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
          1. Search only in the document HEAD (first 6000 chars) to avoid false
             matches from legal definitions deep in T&C body.
          2. 'Vendor'/'Supplier'/'Seller' label → seller
          3. 'Bill To'/'Customer'/'Sold To'/'Ship To' label → buyer
          4. Fallback: first prominent company block = buyer, second = seller
        """
        parties = {
            "buyer":  {"company": None, "phone": None, "gstno": None, "tin": None, "pan": None, "address": None, "email": None},
            "seller": {"company": None, "phone": None, "gstno": None, "tin": None, "pan": None, "address": None, "email": None}
        }

        COMPANY_PATTERN = r'([A-Z][A-Z0-9\s\.,&\-\'\/\(\)]+?)'
        GSTIN_PATTERN   = r'\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}'
        PAN_PATTERN     = r'[A-Z]{5}\d{4}[A-Z]{1}'
        PHONE_PATTERN   = r'\d{10,12}'

        # Only search in the first 6000 characters — buyer/seller blocks are
        # always in the header area. This prevents matching T&C definitions like
        # 'BUYER means Reliance Retail...' that appear deep in legal boilerplate.
        head = text[:6000]

        # ── SELLER: look for Vendor/Supplier/SELLER label ───────────────────
        # Strategy A: single-line label followed immediately by company name
        seller_label_match = re.search(
            rf'(?:Vendor|Supplier)(?!\s+Code)\s+(?!Copy)(?:\n)?\s*{COMPANY_PATTERN}'
            rf'(?:\n|Phone|Address|TIN|GSTIN|P\.O\.|P\.?\s*O\.?\s*Number|$)',
            head, re.IGNORECASE | re.MULTILINE
        )
        if seller_label_match:
            candidate = seller_label_match.group(1).strip()
            bad = {'PURCHASE', 'ORDER', 'INVOICE', 'SUPPLIER', 'VENDOR', 'SELLER', 'AGREES', 'THE', 'AN'}
            if not any(w.upper() in bad for w in candidate.split()):
                parties["seller"]["company"] = candidate
                logger.info(f"Seller by label (A): {parties['seller']['company']}")

        # Strategy B: 'SELLER' keyword at start of a line (may have other columns on same line)
        # (Reliance format: " SELLER                    PURCHASE ORDER\n   Vendor Code : ...\n\n   G.V ENTERPRISES")
        if not parties["seller"]["company"]:
            seller_kw = re.search(r'^\s*SELLER\b', head, re.IGNORECASE | re.MULTILINE)
            if seller_kw:
                # Skip the rest of the SELLER line (may contain "PURCHASE ORDER" etc.)
                rest = head[seller_kw.end():]
                newline_pos = rest.find('\n')
                after = rest[newline_pos + 1:] if newline_pos != -1 else ''
                for line in after.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    # Skip meta lines
                    if re.match(r'(?:Vendor\s*Code|Vendor\s*Status|Attention|PURCHASE|PO\s*(?:NO|Date)|Pin\s*Code|E-Mail|Pan|GSTN)\s*:', stripped, re.IGNORECASE):
                        continue
                    if re.match(r'^[\d\W]+$', stripped):
                        continue
                    if stripped.upper() in ('PURCHASE ORDER', 'SELLER', 'BUYER', 'VENDOR'):
                        continue
                    if re.search(r'[A-Za-z]{3}', stripped) and len(stripped) >= 3:
                        parties["seller"]["company"] = stripped
                        logger.info(f"Seller by label (B): {stripped}")
                        break


        # ── BUYER: look for Bill To / Customer / Sold To / Ship To label ───
        # NOTE: 'Buyer :' footer on Reliance POs is just a code (RROPSFNR), not
        # the company name — so we exclude that pattern.
        buyer_label_match = re.search(
            rf'(?:Bill\s*To|Sold\s*To|Customer|Ship\s*To)\s*\n?\s*{COMPANY_PATTERN}'
            rf'(?:\n|Phone|Address|TIN|GSTIN|$)',
            head, re.IGNORECASE | re.MULTILINE
        )
        if buyer_label_match:
            candidate = buyer_label_match.group(1).strip()
            bad = {'PURCHASE', 'ORDER', 'INVOICE', 'SUPPLIER', 'VENDOR', 'SELLER'}
            if not any(w.upper() in bad for w in candidate.split()):
                parties["buyer"]["company"] = candidate
                logger.info(f"Buyer by label: {parties['buyer']['company']}")

        # ── FALLBACK: extract all company+phone blocks ───────────────────────
        if not parties["buyer"]["company"] or not parties["seller"]["company"]:
            company_blocks = re.findall(
                rf'^(?:.*?\)\s*)?{COMPANY_PATTERN}\s+Phone\s*:\s*({PHONE_PATTERN})',
                head, re.MULTILINE | re.IGNORECASE
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

        # ── LAST-RESORT BUYER: pick the first non-empty line of the doc ─────
        # Handles formats like Reliance where the buyer name is on line 1 with
        # no label (e.g., "Reliance   Retail Limited")
        if not parties["buyer"]["company"]:
            for line in head.splitlines():
                stripped = line.strip()
                # Must look like a real company name: uppercase, ≥5 chars, not a date/number
                if (stripped and len(stripped) >= 5
                        and re.search(r'[A-Za-z]{3}', stripped)
                        and not re.match(r'^[\d\W]+$', stripped)
                        and not any(skip in re.sub(r'\s+', ' ', stripped.upper()) for skip in [
                            'PURCHASE ORDER', 'PO NO', 'PO DATE', 'PAGE', 'DELIVERY', 'SUPPLIER COPY', 'AUTO PO'])):
                    # Strip out trailing Phone/TIN/GSTIN fields
                    comp_name = re.split(r'\s{2,}(?:Phone|TIN|GSTIN|Date|P\.O\.)', stripped, flags=re.IGNORECASE)[0].strip()
                    parties["buyer"]["company"] = comp_name
                    logger.info(f"Buyer by first-line fallback: {comp_name}")
                    break

        # ── Address and Block Metadata extraction for Buyer and Seller (if missing) ────────────
        for party_key in ["buyer", "seller"]:
            if parties[party_key]["company"]:
                # Find the company name in the text
                comp_match = re.search(re.escape(parties[party_key]["company"]), head)
                if comp_match:
                    rest_of_text = head[comp_match.end():]
                    lines = rest_of_text.splitlines()
                    
                    # Construct the block of lines to check: the company line itself + the next 4 lines
                    company_line = head[:comp_match.end()].splitlines()[-1] + lines[0] if lines else ''
                    check_lines = [company_line] + lines[1:5]
                    
                    # 1. Extract metadata from these lines explicitly so it's bound tightly to this party
                    for line in check_lines:
                        if not parties[party_key]["tin"]:
                            tin_match = re.search(rf'(?<![A-Z])TIN\s*(?:No?\.?)?\s*:?\s*({GSTIN_PATTERN})', line, re.IGNORECASE)
                            if tin_match: parties[party_key]["tin"] = tin_match.group(1)
                        if not parties[party_key]["gstno"]:
                            gst_match = re.search(rf'(?:GSTIN|GSTN)\s*(?:No?\.?)?\s*:?\s*({GSTIN_PATTERN})', line, re.IGNORECASE)
                            if gst_match: parties[party_key]["gstno"] = gst_match.group(1)
                        if not parties[party_key]["pan"]:
                            pan_match = re.search(rf'PAN\s*(?:No?\.?)?\s*:?\s*({PAN_PATTERN})', line, re.IGNORECASE)
                            if pan_match: parties[party_key]["pan"] = pan_match.group(1)
                        if not parties[party_key]["phone"]:
                            ph_match = re.search(rf'Phone\s*:\s*({PHONE_PATTERN})', line, re.IGNORECASE)
                            if ph_match: parties[party_key]["phone"] = ph_match.group(1)
                    
                    # 2. Extract the address
                    if not parties[party_key]["address"]:
                        address_parts = []
                        for line in lines[1:5]:
                            stripped = line.strip()
                        if not stripped:
                            continue
                        if re.search(r'(?:Vendor|Delivery|Auto PO|Buyer\s*:|Seller\s*:)', stripped, re.IGNORECASE):
                            break
                        
                        # Skip standalone "Address" label
                        if stripped.upper() in ('ADDRESS', 'ADDRESS :', 'ADDRESS:'):
                            continue
                            
                        # Skip lines that start with metadata labels
                        if re.match(r'^(?:Date|P\.?O\.?\s*(?:Expiry|Number|No)|Phone|TIN|GSTIN|Attention)\s*:', stripped, re.IGNORECASE):
                            continue
                        
                        # strip out TIN/Phone/GSTIN fields that might be on the right side
                        addr_part = re.split(r'\s{2,}(?:Phone|TIN|GSTIN|Date|P\.O\.|Expiry)', stripped, flags=re.IGNORECASE)[0].strip()
                        
                        if addr_part:
                            address_parts.append(addr_part)
                    
                    if address_parts:
                        parties[party_key]["address"] = ", ".join(address_parts)
                        logger.info(f"{party_key.capitalize()} address extracted: {parties[party_key]['address']}")

        # ── Delivery Address block: extract buyer GSTN + email ────────────────
        # Reliance-format POs print the site/DC GSTN and email under the
        # "Delivery Address :" block. These belong to the buyer (the store/DC
        # that will receive the goods).  Extract them directly by regex so the
        # proximity-based logic below doesn't mis-assign them to the seller.
        delivery_block_match = re.search(
            r'Delivery\s+Address\s*:(.{50,2000}?)(?:Total\s+Order\s+Value|Delivery\s+Term|Payment\s+Term|Buyer\s*:)',
            text, re.IGNORECASE | re.DOTALL
        )
        if delivery_block_match:
            block = delivery_block_match.group(1)
            # GSTN in delivery block → buyer.gstno
            if not parties["buyer"]["gstno"]:
                gstn_m = re.search(rf'(?:GSTIN|GSTN)\s*No?\s*:?\s*({GSTIN_PATTERN})', block, re.IGNORECASE)
                if gstn_m:
                    parties["buyer"]["gstno"] = gstn_m.group(1)
                    logger.info(f"Buyer GSTN from Delivery Address block: {parties['buyer']['gstno']}")
            # Email in delivery block → buyer.email
            if not parties["buyer"]["email"]:
                email_m = re.search(r'(?:EMAIL|E-?Mail)\s*:?\s*([\w.\-+]+@[\w.\-]+\.\w+)', block, re.IGNORECASE)
                if email_m:
                    parties["buyer"]["email"] = email_m.group(1)
                    logger.info(f"Buyer email from Delivery Address block: {parties['buyer']['email']}")

        # ── Also extract buyer email from standalone "Email :" near buyer footer ─
        # Reliance footer prints: "For RRL-TN-...\n  Reliance Retail Limited\n  Email : rrho.replenishment@ril.com"
        if not parties["buyer"]["email"]:
            buyer_co = parties["buyer"]["company"] or ""
            # Look for Email label within 500 chars after the buyer company name
            bpos = text.find(buyer_co) if buyer_co else -1
            if bpos != -1:
                nearby = text[bpos: bpos + 600]
                email_m2 = re.search(r'(?:EMAIL|E-?Mail)\s*:?\s*([\w.\-+]+@[\w.\-]+\.\w+)', nearby, re.IGNORECASE)
                if email_m2:
                    parties["buyer"]["email"] = email_m2.group(1)
                    logger.info(f"Buyer email from footer block: {parties['buyer']['email']}")

        # ── GSTIN extraction (only GSTIN/GSTN labels → gstno) ──────────────────
        all_gstins = re.findall(
            rf'(?:GSTIN|GSTN)\s*(?:No?\.?)?\s*:?\s*({GSTIN_PATTERN})',
            text, re.IGNORECASE
        )

        if len(all_gstins) >= 1:
            for gstin_val in all_gstins:
                if gstin_val == parties["buyer"]["gstno"] or gstin_val == parties["seller"]["gstno"]:
                    continue
                pos = text.find(gstin_val)
                buyer_pos  = text.find(parties["buyer"]["company"])  if parties["buyer"]["company"]  else -1
                seller_pos = text.find(parties["seller"]["company"]) if parties["seller"]["company"] else -1

                if buyer_pos != -1 and seller_pos != -1:
                    if abs(pos - buyer_pos) < abs(pos - seller_pos):
                        if not parties["buyer"]["gstno"]: parties["buyer"]["gstno"] = gstin_val
                    else:
                        if not parties["seller"]["gstno"]: parties["seller"]["gstno"] = gstin_val
                elif buyer_pos != -1 and not parties["buyer"]["gstno"]: parties["buyer"]["gstno"] = gstin_val
                elif seller_pos != -1 and not parties["seller"]["gstno"]: parties["seller"]["gstno"] = gstin_val

        # ── PAN extraction (only PAN label → pan) ────────────────────────────
        all_pans = re.findall(
            rf'PAN\s*(?:No?\.?)?\s*:?\s*({PAN_PATTERN})',
            text, re.IGNORECASE
        )

        if len(all_pans) >= 1:
            for pan_val in all_pans:
                if pan_val == parties["buyer"]["pan"] or pan_val == parties["seller"]["pan"]:
                    continue
                pos = text.find(pan_val)
                buyer_pos  = text.find(parties["buyer"]["company"])  if parties["buyer"]["company"]  else -1
                seller_pos = text.find(parties["seller"]["company"]) if parties["seller"]["company"] else -1

                if buyer_pos != -1 and seller_pos != -1:
                    if abs(pos - buyer_pos) < abs(pos - seller_pos):
                        if not parties["buyer"]["pan"]: parties["buyer"]["pan"] = pan_val
                    else:
                        if not parties["seller"]["pan"]: parties["seller"]["pan"] = pan_val
                elif buyer_pos != -1 and not parties["buyer"]["pan"]: parties["buyer"]["pan"] = pan_val
                elif seller_pos != -1 and not parties["seller"]["pan"]: parties["seller"]["pan"] = pan_val

        # ── TIN extraction (only TIN label → tin) ────────────────────────────
        all_tins = re.findall(
            rf'(?<![A-Z])TIN\s*(?:No?\.?)?\s*:?\s*({GSTIN_PATTERN})',
            text, re.IGNORECASE
        )

        if len(all_tins) >= 1:
            for tin_val in all_tins:
                if tin_val == parties["buyer"]["tin"] or tin_val == parties["seller"]["tin"]:
                    continue
                pos = text.find(tin_val)
                buyer_pos  = text.find(parties["buyer"]["company"])  if parties["buyer"]["company"]  else -1
                seller_pos = text.find(parties["seller"]["company"]) if parties["seller"]["company"] else -1

                if buyer_pos != -1 and seller_pos != -1:
                    if abs(pos - buyer_pos) < abs(pos - seller_pos):
                        if not parties["buyer"]["tin"]: parties["buyer"]["tin"] = tin_val
                    else:
                        if not parties["seller"]["tin"]: parties["seller"]["tin"] = tin_val
                elif buyer_pos != -1 and not parties["buyer"]["tin"]: parties["buyer"]["tin"] = tin_val
                elif seller_pos != -1 and not parties["seller"]["tin"]: parties["seller"]["tin"] = tin_val

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
        Build the CONFIRMED VALUES section to inject into the Qwen prompt.
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
        if parties["buyer"].get("address"):
            lines.append(f"- buyer.address: {parties['buyer']['address']}")
        if parties["buyer"].get("gstno"):
            lines.append(f"- buyer.gstno: {parties['buyer']['gstno']}")
        if parties["buyer"].get("tin"):
            lines.append(f"- buyer.tin: {parties['buyer']['tin']}")
        if parties["buyer"].get("pan"):
            lines.append(f"- buyer.pan: {parties['buyer']['pan']}")
        if parties["buyer"].get("email"):
            lines.append(f"- buyer.email: {parties['buyer']['email']}")
        if parties["buyer"].get("phone"):
            lines.append(f"- buyer.phone: {parties['buyer']['phone']}")
        if parties["seller"].get("company"):
            lines.append(f"- seller.companyName: {parties['seller']['company']}")
        if parties["seller"].get("address"):
            lines.append(f"- seller.address: {parties['seller']['address']}")
        if parties["seller"].get("gstno"):
            lines.append(f"- seller.gstno: {parties['seller']['gstno']}")
        if parties["seller"].get("tin"):
            lines.append(f"- seller.tin: {parties['seller']['tin']}")
        if parties["seller"].get("pan"):
            lines.append(f"- seller.pan: {parties['seller']['pan']}")
        if parties["seller"].get("email"):
            lines.append(f"- seller.email: {parties['seller']['email']}")
        if parties["seller"].get("phone"):
            lines.append(f"- seller.phone: {parties['seller']['phone']}")

        return "\n".join(lines)

