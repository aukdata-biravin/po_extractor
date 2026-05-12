"""
AI-based parsing of extracted text to Purchase Order JSON using NVIDIA API with Mistral model.
"""

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import ValidationError

try:
    import json5
    HAS_JSON5 = True
except ImportError:
    HAS_JSON5 = False

from config import settings
from schemas.purchase_order import PurchaseOrder

# ── Token-budget constants ────────────────────────────────────────────────────
# The NVIDIA Mixtral endpoint enforces a 64 k *total* context window
# (input tokens + output tokens combined).  We keep a safety margin so we
# never send a request that would exceed the limit.
_MODEL_CONTEXT_LIMIT = 63_000  # leave 1 k headroom below the hard 64 k cap
_CHARS_PER_TOKEN     = 3.5     # conservative character-to-token ratio for English

logger = logging.getLogger(__name__)


class AIParser:
    """Parses extracted text using NVIDIA API with Mistral model."""

    def __init__(self) -> None:
        """
        Initialise AI Parser from the global settings singleton.
        No arguments needed — API key and model are read from *config.settings*.
        """
        self.model = settings.model_name
        self.client = OpenAI(
            base_url=settings.api_base_url,
            api_key=settings.nvidia_api_key,
        )
        logger.info("Initialised NVIDIA API client (model=%s)", self.model)

    # ── Token-budget helper ───────────────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Return a conservative *upper-bound* token estimate for *text*."""
        return int(len(text) / _CHARS_PER_TOKEN) + 1
    
    def _call_api(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
        """
        Make a chat-completion API call and return the cleaned JSON text.

        The method dynamically caps *max_tokens* so that the estimated input
        token count + requested output tokens never exceed the model's context
        window.  This prevents the ``generation exceeded max tokens limit``
        error that occurs when large documents are sent alongside a high
        max_tokens value.
        """
        input_tokens = (
            self._estimate_tokens(system_prompt)
            + self._estimate_tokens(user_prompt)
        )
        available_for_output = _MODEL_CONTEXT_LIMIT - input_tokens

        if available_for_output <= 0:
            raise ValueError(
                f"Input text is too large for the model context window "
                f"(estimated ~{input_tokens:,} tokens vs {_MODEL_CONTEXT_LIMIT:,} limit). "
                "Please reduce the document size or split it into pages."
            )

        safe_max_tokens = min(max_tokens, available_for_output)
        if safe_max_tokens < max_tokens:
            logger.warning(
                "max_tokens capped from %d → %d (input ~%d tokens, context limit %d)",
                max_tokens, safe_max_tokens, input_tokens, _MODEL_CONTEXT_LIMIT,
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=safe_max_tokens,
            top_p=1,
        )
        response_text = response.choices[0].message.content or ""
        logger.info("API response received (%d chars, requested %d output tokens)",
                    len(response_text), safe_max_tokens)
        return self._extract_json_from_response(response_text)

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract clean JSON string from a model response (handles markdown fences, comments, prose)."""
        response_text = response_text.strip()

        # Search for ```json fence anywhere (model may add prose before it)
        fence_idx = response_text.find("```json")
        if fence_idx != -1:
            json_start = response_text.find("\n", fence_idx + 7)
            if json_start == -1:
                json_start = fence_idx + 7
            else:
                json_start += 1
            json_end = response_text.rfind("```")
            if json_end > json_start:
                response_text = response_text[json_start:json_end].strip()
            else:
                response_text = response_text[json_start:].strip()
        elif response_text.startswith("```"):
            json_start = response_text.find("\n", 3)
            if json_start == -1:
                json_start = 3
            else:
                json_start += 1
            json_end = response_text.rfind("```")
            if json_end > json_start:
                response_text = response_text[json_start:json_end].strip()
            else:
                response_text = response_text[json_start:].strip()
        else:
            # No fence — find first { to skip any prose preamble
            brace_idx = response_text.find("{")
            if brace_idx != -1:
                response_text = response_text[brace_idx:].strip()

        # Strip JS-style // comments (invalid in JSON)
        response_text = re.sub(r'//[^\n]*', '', response_text)
        # Strip /* ... */ block comments
        response_text = re.sub(r'/\*.*?\*/', '', response_text, flags=re.DOTALL)
        # Strip ellipsis placeholders like "...", inside arrays
        response_text = re.sub(r',\s*\.\.\.\s*(?=[}\]])', '', response_text)
        response_text = re.sub(r'\.\.\.\s*,', '', response_text)
        response_text = re.sub(r'\.\.\.',  '', response_text)

        return response_text.strip()

    def _parse_json(self, text: str, context: str = "") -> dict:
        """Parse JSON with fallback to json5 and repair."""
        try:
            data = json.loads(text)
            logger.info(f"Successfully parsed JSON ({context})")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Standard JSON parse failed ({context}): {e}")

        if HAS_JSON5:
            try:
                data = json5.loads(text)
                logger.info(f"Successfully parsed with json5 ({context})")
                return data
            except Exception as e5:
                logger.warning(f"json5 parse failed ({context}): {e5}")

        logger.info(f"Attempting JSON repair ({context})...")
        repaired = self._repair_json(text)
        try:
            data = json.loads(repaired)
            logger.info(f"Successfully parsed after repair ({context})")
            return data
        except json.JSONDecodeError as err:
            # Persist debug artefacts to the system temp dir (writable everywhere)
            debug_dir = Path(tempfile.gettempdir()) / "po_extraction_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "debug_original_error.json").write_text(text, encoding="utf-8")
            (debug_dir / "debug_cleaned_error.json").write_text(repaired, encoding="utf-8")
            logger.error(
                "All JSON parse attempts failed (%s): %s  [debug files → %s]",
                context, err, debug_dir,
            )
            raise

    def parse_po_text(
        self,
        extracted_text: str,
        confirmed_section: str = ""
    ) -> PurchaseOrder:
        """
        Parse extracted PDF text using a two-pass approach:
          Pass 1: Extract header fields (poNumber, poDate, buyer, seller, financials, etc.)
          Pass 2: Extract ALL line items (focused call — avoids model truncation)

        Args:
            extracted_text: Text extracted from PDF using PyMuPDF
            confirmed_section: Pre-formatted string of confirmed extracted values

        Returns:
            PurchaseOrder object with extracted data
        """
        try:
            system_prompt = self._get_system_prompt()

            # ── PASS 1: Header + financials ──────────────────────────────────────
            logger.info("Pass 1: Extracting header / financial summary...")
            header_prompt = self._get_header_prompt(
                extracted_text,
                confirmed_section=confirmed_section
            )
            raw_header = self._call_api(
                system_prompt, header_prompt, max_tokens=settings.header_max_tokens
            )
            header_data = self._parse_json(raw_header, context="header")
            if "purchaseOrder" in header_data:
                header_data = header_data["purchaseOrder"]

            # ── PASS 2: Line items ────────────────────────────────────────────────
            logger.info("Pass 2: Extracting ALL line items...")
            items_prompt = self._get_items_prompt(extracted_text)
            raw_items = self._call_api(
                system_prompt, items_prompt, max_tokens=settings.items_max_tokens
            )

            items_data = self._parse_json(raw_items, context="items")
            # Accept either {"items": [...]} or [...]
            if isinstance(items_data, dict):
                if "items" in items_data:
                    items_list = items_data["items"]
                elif "purchaseOrder" in items_data:
                    items_list = items_data["purchaseOrder"].get("items", [])
                else:
                    items_list = []
            elif isinstance(items_data, list):
                items_list = items_data
            else:
                items_list = []

            logger.info(f"Pass 2 extracted {len(items_list)} line items")

            # ── Merge results ─────────────────────────────────────────────────────
            header_data["items"] = items_list
            # Clean whitespace from all string fields before schema validation.
            # Fixes issues like extra spaces ("PAZHAMUDIR      NILAYAM") and
            # embedded newlines in company names, addresses, etc.
            header_data = self._clean_strings(header_data)
            purchase_order = PurchaseOrder(**header_data)

            logger.info("Successfully parsed Purchase Order with NVIDIA Mistral API (two-pass)")
            return purchase_order

        except ValidationError as e:
            logger.error(f"Failed to validate Purchase Order schema: {e}")
            raise ValueError(f"Invalid Purchase Order structure: {e}")
        except Exception as e:
            logger.error(f"Error parsing PO text with NVIDIA API: {e}")
            raise ValueError(f"Failed to parse Purchase Order: {e}")


    @staticmethod
    def _clean_strings(obj):
        """
        Recursively walk a parsed JSON structure and normalise every string value:
          - Replace any run of whitespace characters (spaces, tabs, \n, \r) with a single space.
          - Strip leading / trailing whitespace.
        Non-string values (numbers, None, lists, dicts) are left untouched.
        """
        if isinstance(obj, dict):
            return {k: AIParser._clean_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [AIParser._clean_strings(item) for item in obj]
        if isinstance(obj, str):
            # Collapse all whitespace sequences (spaces, tabs, newlines) to one space
            cleaned = re.sub(r'\s+', ' ', obj).strip()
            return cleaned if cleaned else None   # empty string → null
        return obj

    def _repair_json(self, json_text: str) -> str:
        """Repair malformed JSON by removing trailing commas and fixing common issues."""
        # First, handle incomplete JSON by checking for open braces/brackets
        # Count opening and closing braces/brackets
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        open_brackets = json_text.count('[')
        close_brackets = json_text.count(']')
        
        # Close any incomplete/truncated string values
        # If the JSON ends mid-string, close it properly
        json_text = json_text.rstrip()
        if not json_text.endswith(('}', ']', '"')):
            # Incomplete string, close it
            json_text += '"'
            logger.warning("Closed incomplete string value at end")
        
        # Add missing closing braces/brackets
        brace_diff = open_braces - close_braces
        bracket_diff = open_brackets - close_brackets
        
        if brace_diff > 0:
            json_text = json_text + ('}' * brace_diff)
            logger.warning(f"Added {brace_diff} missing closing braces")
        
        if bracket_diff > 0:
            json_text = json_text + (']' * bracket_diff)
            logger.warning(f"Added {bracket_diff} missing closing brackets")
        
        # Remove all types of trailing commas
        # Pattern 1: ,} with any whitespace
        json_text = re.sub(r',\s*}', '}', json_text)
        # Pattern 2: ,] with any whitespace  
        json_text = re.sub(r',\s*]', ']', json_text)
        # Pattern 3: Multiple trailing commas
        json_text = re.sub(r'(,\s*)+}', '}', json_text)
        json_text = re.sub(r'(,\s*)+]', ']', json_text)
        
        # Fix common issues with null values followed by commas
        # Handles: null, where next is } or ]
        json_text = re.sub(r'null\s*,\s*([}\]])', r'null\1', json_text)
        
        # Fix missing commas between object properties
        # Handles: "key": value "nextkey" -> "key": value, "nextkey"
        json_text = re.sub(r'("\s*:\s*[^,}\]]+)\s+(")', r'\1, \2', json_text)
        
        # Fix incomplete string values (truncated strings)
        # Look for "key": "incomplete and close them
        json_text = re.sub(r'(":\s*)"([^"]*?)(\s*[,}\]])', r'\1"\2"\3', json_text)
        
        return json_text
    
    def validate_line_items(self, purchase_order: PurchaseOrder) -> dict:
        """
        Validate extracted line items using business rules.
        Safe for string-based schema.
        """
        errors = []
        warnings = []
        
        if not purchase_order.items:
            return {"errors": errors, "warnings": warnings, "itemCount": 0}
            
        def safe_float(val):
            if val is None: return None
            try:
                # Remove commas and non-numeric chars (except dot)
                clean_val = str(val).replace(',', '').strip()
                return float(clean_val)
            except:
                return None
        
        for idx, item in enumerate(purchase_order.items, 1):
            try:
                q_each = safe_float(item.quantityEach)
                q_carton = safe_float(item.quantityCarton)
                mrp_each = safe_float(item.mrpeach)
                base_price = safe_float(item.basicCostPrice)
                landing = safe_float(item.landingRate)
                total_base = safe_float(item.totalBaseValue)
                cgst_p = safe_float(item.cgstPercent)
                
                # 1. Check if Qty is positive
                qty = q_each if q_each is not None else q_carton
                if qty is not None:
                    if qty <= 0:
                        warnings.append(f"Sr{item.srNo}: Quantity {qty} must be positive")
                
                # 2. Check Total = Rate × Qty
                if (base_price is not None and q_carton is not None and total_base is not None):
                    expected_total = round(base_price * q_carton, 2)
                    if abs(expected_total - total_base) > 2.0:
                        warnings.append(
                            f"Sr{item.srNo}: Total mismatch. "
                            f"Expected {expected_total} (Cost {base_price} × Qty {q_carton}) "
                            f"but got {total_base}"
                        )
                
                # 3. Check pricing hierarchy
                if (mrp_each is not None and base_price is not None):
                    if mrp_each < base_price:
                        warnings.append(
                            f"Sr{item.srNo}: MRP {mrp_each} < Basic Cost {base_price}"
                        )
                
                if (landing is not None and base_price is not None):
                    if landing < base_price:
                        warnings.append(
                            f"Sr{item.srNo}: Landing Rate {landing} < Basic Cost {base_price}"
                        )
            except Exception as e:
                warnings.append(f"Sr{getattr(item, 'srNo', idx)}: Validation error - {str(e)}")
                
        return {
            "errors": errors, 
            "warnings": warnings, 
            "itemCount": len(purchase_order.items)
        }
    
    def _get_system_prompt(self) -> str:
        return """Extract structured JSON from the provided purchase order document.

        GOLDEN RULES:
        - Return ONLY valid JSON. No preamble, no markdown fences, no extra text.
        - NEVER CALCULATE. NEVER FORMAT. Extract ONLY what is physically printed in the document.
        - Copy numeric values exactly as printed — do not strip commas, do not round, do not reformat.
        Example: if the document shows "61,506" extract "61,506". If it shows "2,340.00" extract "2,340.00".
        - If a value is not present in the document, return null.
        - If a value is printed as 0 or 0.00, extract it as-is — do not treat it as null.
        - Do not derive, infer, or compute any field from other fields.
        - Do not strip labels — extract the raw value only (e.g., GSTIN value, not "GSTIN: 33XXXX").

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        PURCHASE ORDER HEADER
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        purchaseOrder.poNumber
        - Extract the PO Number / Purchase Order Number / P.O. Number / Buyer Order Number exactly as printed.

        purchaseOrder.poDate
        - Extract Purchase Order Date / PO Date / Date exactly as printed.

        purchaseOrder.expiryDate
        - Extract Expiry Date / P.O. Expiry / Valid Until exactly as printed, else null.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        BUYER (the entity issuing / receiving the PO)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        IDENTIFICATION RULE:
        - The BUYER is the company or store placing the order.
        - In formal enterprise POs, the buyer name and address appear in the header or letterhead.
        - In smaller/retail-format POs, the buyer may be a store name printed prominently at the top
        with an address block — treat that as the buyer.
        - Do NOT use the "Delivery Address" or "Ship To" address as the buyer address.

        buyer.companyName
        - Extract the buyer/store/company name exactly as printed.

        buyer.address
        - Extract the address printed directly below the buyer company name in the document header or letterhead only. 
        - Never use blocks labeled "Delivery Address", "Ship To", or "Distribution Center" for this field even if they appear on the same page.

        buyer.gstin
        - Extract the buyer GSTIN / TIN / GSTN exactly as printed (value only, no label).
        - If not found in the main header, look for a GSTN/GSTIN in the "Delivery Address" or "GSTIN Number Details" section and use that as the buyer GSTIN.

        buyer.email
        - Extract buyer email exactly as printed, else null.

        buyer.phone
        - Extract buyer phone/mobile exactly as printed, else null.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        SELLER (the vendor/supplier fulfilling the PO)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        IDENTIFICATION RULE:
        - The SELLER is the vendor/supplier delivering the goods.
        - Typically in a "Vendor" / "Seller" / "Supplier" section of the document.

        seller.companyName
        - Extract seller/vendor company name exactly as printed.

        seller.address
        - Extract seller/vendor address exactly as printed.

        seller.gstin
        - Extract seller GSTIN / TIN exactly as printed (value only, no label).

        seller.email
        - Extract seller email exactly as printed, else null.

        seller.phone
        - Extract seller phone/mobile exactly as printed, else null.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        FINANCIAL SUMMARY
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        RULE: Extract only from the summary/footer/totals section of the document.
        Do NOT sum item-level values. Do NOT calculate. Extract only what is printed.

        financialSummary.totalBasicValue
        - Extract Total Basic Value / Subtotal / Taxable Amount exactly as printed, else null.

        financialSummary.totalCGST
        - Extract Total CGST amount exactly as printed, else null.

        financialSummary.totalSGST
        - Extract Total SGST amount exactly as printed, else null.

        financialSummary.totalIGST
        - Extract Total IGST amount exactly as printed, else null.

        financialSummary.totalGST
        - Extract a combined Total GST figure only if printed as a single labeled total, else null.
        - Do NOT add CGST + SGST to populate this field.

        financialSummary.totalOrderValue
        - Extract the grand total / Net Total / Total Order Value / Total Amount exactly as printed.

        financialSummary.discountPercent
        - Extract overall discount percentage exactly as printed, else null.

        financialSummary.discountAmt
        - Extract overall discount amount exactly as printed, else null.

        financialSummary.totalQuantity
        - Extract total quantity from the summary/footer row exactly as printed.
        - It may be labeled: "Total Qty", "Grand Total of Qty", "Total Units", or appear unlabeled
        as a number positioned under the Qty column in a totals row (e.g., "NET TOTAL  174  3640.76"
        — the value under the Qty column is 174).
        - Do NOT confuse with the grand total amount.

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        LINE ITEMS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        GENERAL RULES:
        - Extract each line item independently. Do not merge separate items.
        - A single item may span multiple text rows — collect all parts before mapping.
        - Extract all values exactly as printed. Do not reformat, round, or calculate.
        - ZERO VALUES ARE DATA: If a value is 0, 0.0, or 0.00 in the document, you MUST extract it exactly as it appears. Do NOT treat 0 as null.
        - EXAMPLE FOR VERTICAL TAX STACKING RULE (RELIANCE FORMAT): 
          One item occupies a block of 4 physical rows. The column headers for the last two columns are:
            CGST(%) | CGST
            SGST(%) | SGST
            CESS(%) | CESS
            CessFxdRt | CessFxdVl
          These headers do NOT repeat per item. Map by row position within each item block.

          EXAMPLE — Item 1 in the document:
            Row 1: 1  490001912  8906001050490  MOTHER RECP AP GINGER PCKLE 300G  1.000  CAR  3,150.00  2,340.00      2.50    58.50   2,340.00
            Row 2:    20019000                  25.11.2025                                                             2.50    58.50
            Row 3:                              SAHO                                                                   0.00     0.00
            Row 4:                                                                 30.000   EA   105.00               0.00     0.00

          MANDATORY MAPPING for this item:
            srNo=1, articleCode=490001912, eanCode=8906001050490
            productDescription="MOTHER RECP AP GINGER PCKLE 300G"
            quantityCarton="1.000", uomCarton="CAR", mrpcarton="3,150.00"
            basicCostPrice="2,340.00", totalBaseValue="2,340.00"
            cgstPercent="2.50", cgstAmt="58.50"   ← from Row 1 tax columns
            sgstPercent="2.50", sgstAmt="58.50"   ← from Row 2 tax columns
            cessPercent="0.00", cessAmt="0.00"    ← from Row 3 tax columns
            cessFxdRt="0.00", cessFxdVl="0.00"   ← from Row 4 tax columns (CRITICAL: these must NOT be null!)
            quantityEach="30.000", uomEach="EA", mrpeach="105.00"

          KEY RULE: Even if cessFxdRt and cessFxdVl are "0.00", you MUST extract them as "0.00". NEVER return null for these fields if a row exists with two numbers in the tax column area.

        SPLIT-ROW ITEM ASSEMBLY RULE (critical for this document):
        Some PDFs render a single line item across TWO consecutive extracted text rows because
        the product description overflows and the PDF renderer wraps it to the next row:

          Row A:  <srNo>  [no articleCode]  [no description]  <MRP>  <cost>  <gst%>  <rate>  <qty>  <total>
          Row B:          <articleCode>      <full description>

        These two rows belong to the SAME item. Assemble them as one:
          - srNo             → from Row A
          - articleCode      → from Row B
          - productDescription → full text from Row B (plus any further continuation rows)
          - All numeric fields (MRP, basicCostPrice, gstPercent, landingRate, quantityEach, totalValueWithTax)
            → from Row A

        DETECTION SIGNAL: A row has numeric price/qty/total values but is missing articleCode
        AND the very next row begins with a numeric code followed by text but has NO numerics.
        → Those two rows are one item.

        IMPORTANT:
        - VERTICAL TAX STACKING RULE: In some documents (e.g., Reliance), tax columns like CGST, SGST, CESS, and Fixed Rates are stacked vertically within the same item block across multiple physical lines. You must read all lines belonging to an item and map the corresponding values based on their horizontal alignment with the column headers.
        - Do NOT create a separate item for Row B alone.
        - Do NOT create an item from Row B's numbers (it has none).
        - Do NOT skip Row A's numeric values.
        - Do NOT merge two genuinely different items that both have their own srNo and numbers.

        srNo
        - Extract serial/item/line number exactly as printed.

        articleCode
        - Extract article code / item code / product code exactly as printed.

        hsnCode
        - Extract HSN / SAC code exactly as printed, else null.
        - STRICT: Do NOT map quantity, EAN, or article codes here.

        eanCode
        - Extract EAN / barcode / UPC exactly as printed, else null.

        vendorArticleNo
        - Extract vendor article number exactly as printed if present as a distinct column, else null.

        vendorItemNo
        - Extract vendor item number exactly as printed if present as a distinct column, else null.

        productDescription
        - Extract full product description / material description exactly as printed.
        - If it wraps across multiple rows, extract the complete text.

        deliveryDate
        - Extract item-level delivery date exactly as printed, else null.

        ─── QUANTITY & UOM ──────────────────────────────────────────

        quantityCarton
        - Extract carton/box/case quantity exactly as printed if a carton UOM row exists.

        uomCarton
        - Extract the UOM label for the carton row exactly as printed (e.g., CAR, C01, C02, CV, E01, E02).

        quantityEach
        - Extract the each/unit/individual quantity exactly as printed.
        - If the PO has only a single quantity column (no carton/each split), put that value here.
        - If the PO has separate carton and each rows, put the each/unit row value here.

        uomEach
        - Extract the UOM label for the each/unit row exactly as printed (e.g., EA, Nos, Pcs).
        ─── MRP ─────────────────────────────────────────────────────

        mrpcarton
        - Extract MRP printed for the carton/case row exactly as printed, else null.

        mrpeach
        - Extract the MRP printed for the each/unit row.
        - If the PO has only a single MRP column (no dual rows), put that value here.

        ─── PRICING ─────────────────────────────────────────────────

        basicCostPrice
        - Extract the unit cost/rate BEFORE tax exactly as printed.
        - Labeled: Base Cost, Base Cost Price, Unit Price, Rate, Basic Cost Price.
        - Extract the per-unit/per-carton value exactly as it appears — do not multiply.

        landingRate
        - Extract the unit rate AFTER tax exactly as printed if that column exists, else null.
        - Labeled: Landing Rate, Rate incl. tax.

        ─── TAX ─────────────────────────────────────────────────────

        TAX COLUMN FORMAT VARIANTS:
        Variant A — Stacked columns (e.g., Reliance format):
            Percentage rows: CGST(%) / SGST(%)
            Amount rows:     CGST amt / SGST amt
            Map each printed value to its field by vertical position.

        Variant B — Single GST column (e.g., retail/auto-PO format):
            One "GST %" column and one GST amount column per item.
            Map to gstPercent and gstAmt only.
            Leave cgstPercent, sgstPercent, cgstAmt, sgstAmt as null.

        gstPercent
        - Extract combined GST % exactly as printed ONLY if a single "GST %" column exists.
        - Do NOT populate from CGST% or SGST% values.

        cgstPercent
        - Extract CGST % exactly as printed if a CGST(%) column exists, else null.

        sgstPercent
        - Extract SGST % exactly as printed if an SGST(%) column exists, else null.

        igstPercent
        - Extract IGST % exactly as printed if present, else null.

        cgstAmt
        - Extract CGST amount exactly as printed if present, else null.

        sgstAmt
        - Extract SGST amount exactly as printed if present, else null.

        igstAmt
        - Extract IGST amount exactly as printed if present, else null.

        gstAmt
        - Extract combined GST amount exactly as printed ONLY if a single GST amount column exists.
        - Do NOT add cgstAmt + sgstAmt to populate this.

        cessPercent
        - Extract CESS percentage exactly as printed. 
        - IMPORTANT: In complex layouts (like Reliance), CESS% often appears on a separate line BELOW the CGST/SGST lines in the same item block. Look vertically down the tax columns.
        - Extract 0.00 if explicitly printed; only return null if the cell is truly blank.

        cessAmt
        - Extract CESS amount exactly as printed. 
        - Look vertically down the tax columns if not on the main line.
        - Extract 0.00 if explicitly printed.

        cessFxdVl
        - Extract CESS Fixed Value exactly as printed if present (often at the bottom of the tax column stack), else null.

        cessFxdRt
        - Extract CESS Fixed Rate exactly as printed if present (often at the bottom of the tax column stack), else null.

        totalTaxAmt
        - Extract total tax amount exactly as printed ONLY if it appears as a labeled sum in the item row.
        - Do NOT calculate. If not explicitly printed, return null.

        totalBaseValue
        - Extract taxable/base amount for the line item exactly as printed if present, else null.
        - Do NOT calculate.

        totalValueWithTax
        - Extract the final line item total inclusive of tax exactly as printed if present, else null.
        - Labeled: Total Amount, Amount, Line Total.
        - Do NOT calculate. If not explicitly printed, return null.

        discountPercent
        - Extract item-level discount percentage exactly as printed, else null.

        discountAmt
        - Extract item-level discount amount exactly as printed, else null.

        Return JSON strictly in the schema structure provided.
        """
    
    def _get_header_prompt(
        self,
        extracted_text: str,
        confirmed_section: str = ""
    ) -> str:
        """Get user prompt for header/financial extraction only (no items array)."""
        schema = {
            "purchaseOrder": {
                "poNumber": "string or null",
                "poDate": "string or null",
                "expiryDate": "string or null",
                "buyer": {
                    "companyName": "string or null",
                    "address": "string or null",
                    "gstin": "string or null",
                    "email": "string or null",
                    "phone": "string or null"
                },
                "seller": {
                    "companyName": "string or null",
                    "address": "string or null",
                    "email": "string or null",
                    "gstin": "string or null",
                    "phone": "string or null"
                },
                "financialSummary": {
                    "totalBasicValue": "string or null",
                    "totalCGST": "string or null",
                    "totalSGST": "string or null",
                    "totalIGST": "string or null",
                    "totalGST": "string or null",
                    "totalOrderValue": "string or null",
                    "totalQuantity": "string or null",
                    "discountPercent": "string or null",
                    "discountAmt": "string or null"
                }
            }
        }

        confirmed_block = f"\n\n{confirmed_section}\n" if confirmed_section else ""

        return f"""Read the following document text and fill in the JSON schema below.
The text layout uses spaces to preserve the original document structure — values on the same line may be side-by-side columns.

<DOCUMENT>
{extracted_text}
</DOCUMENT>{confirmed_block}

Fill this schema (header fields only, no items):

<SCHEMA>
{json.dumps(schema, indent=2)}
</SCHEMA>

Special rule for financialSummary.totalQuantity:
- Look for a summary/footer row (often on the LAST page). It may be labeled "Grand Total of Qty", "NET TOTAL", "GRAND TOTAL", "Total", or similar.
- That row often contains TWO numbers: one positioned under the Qty/Quantity column and one under the Amount/Total column.
- The number under the Qty column = totalQuantity. The number under the Amount column = totalOrderValue.
- Extract totalQuantity even when it has NO explicit label — rely on its column position.

Return only valid JSON. Missing values = null."""

    def _get_items_prompt(self, extracted_text: str) -> str:
        """Get user prompt for extracting ALL line items only."""
        item_schema = {
            "srNo": "string or null",
            "articleCode": "string or null",
            "hsnCode": "string or null",
            "eanCode": "string or null",
            "productDescription": "string or null",
            "deliveryDate": "string or null",
            "quantityCarton": "string or null",
            "quantityEach": "string or null",
            "uomCarton": "string or null",
            "uomEach": "string or null",
            "mrpcarton": "string or null",
            "mrpeach": "string or null",
            "basicCostPrice": "string or null",
            "landingRate": "string or null",
            "gstPercent": "string or null",
            "cgstPercent": "string or null",
            "sgstPercent": "string or null",
            "igstPercent": "string or null",
            "cgstAmt": "string or null",
            "sgstAmt": "string or null",
            "igstAmt": "string or null",
            "gstAmt": "string or null",
            "cessPercent": "string or null",
            "cessAmt": "string or null",
            "cessFxdVl": "string or null",
            "cessFxdRt": "string or null",
            "totalTaxAmt": "string or null",
            "totalBaseValue": "string or null",
            "totalValueWithTax": "string or null",
            "discountPercent": "string or null",
            "discountAmt": "string or null"
        }

        return f"""Read the document below and extract every line item from the product/items table.

<DOCUMENT>
{extracted_text}
</DOCUMENT>

Return a JSON object with a single "items" array containing every row. Missing cell = null. No truncation.

<SCHEMA>
{{
  "items": [
    {json.dumps(item_schema, indent=4)}
  ]
}}
</SCHEMA>

Return only valid JSON."""

