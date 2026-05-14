"""
AI-based parsing of extracted text to Purchase Order JSON using NVIDIA API with Qwen model.
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
# Qwen3.5 supports up to 262k tokens; we cap conservatively at 100k for safety.
_MODEL_CONTEXT_LIMIT = 100_000
_CHARS_PER_TOKEN     = 3.5     # conservative character-to-token ratio for English

logger = logging.getLogger(__name__)


class AIParser:
    """Parses extracted text using NVIDIA API with Qwen model."""

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
            extra_body={"chat_template_kwargs": {"thinking": False}},
        )
        msg = response.choices[0].message
        # Qwen3 may put its answer in reasoning_content when thinking is active;
        # fall back to that if content is empty.
        response_text = (msg.content or "").strip()
        if not response_text:
            response_text = getattr(msg, "reasoning_content", "") or ""
            if response_text:
                logger.warning("message.content was empty — using reasoning_content fallback")

        # Strip <think>...</think> blocks produced by chain-of-thought reasoning
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        if not response_text:
            raise ValueError(
                "Model returned an empty response. "
                "The document may be too large or the model is not available."
            )

        logger.info("API response received (%d chars, requested %d output tokens)",
                    len(response_text), safe_max_tokens)
        return self._extract_json_from_response(response_text)

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract clean JSON string from a model response (handles markdown fences, comments, prose)."""
        response_text = response_text.strip()

        # 1. Try markdown fences first
        fence_patterns = [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]
        for pattern in fence_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                response_text = match.group(1).strip()
                break
        else:
            # 2. No fence — find the outermost { } or [ ]
            # Find first occurrence of { or [
            start_idx_brace = response_text.find("{")
            start_idx_bracket = response_text.find("[")
            
            if start_idx_brace != -1 and (start_idx_bracket == -1 or start_idx_brace < start_idx_bracket):
                # Starts with a brace
                start_idx = start_idx_brace
                end_idx = response_text.rfind("}")
            elif start_idx_bracket != -1:
                # Starts with a bracket
                start_idx = start_idx_bracket
                end_idx = response_text.rfind("]")
            else:
                start_idx = -1
                end_idx = -1

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx + 1].strip()

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
            # Log the first 200 chars to help identify the issue in the terminal
            snippet = text[:200].replace('\n', ' ')
            logger.error(f"Failed JSON snippet ({context}): {snippet}...")
            raise

    def parse_po_text(
        self,
        extracted_text: str,
        confirmed_section: str = ""
    ) -> PurchaseOrder:
        """
        Parse extracted PDF text using a single-pass approach.

        Args:
            extracted_text: Text extracted from PDF using PyMuPDF
            confirmed_section: Pre-formatted string of confirmed extracted values

        Returns:
            PurchaseOrder object with extracted data
        """
        try:
            system_prompt = self._get_system_prompt()

            # ── SINGLE PASS: All fields + line items ─────────────────────────────
            logger.info("Extracting PO data in a single pass...")
            user_prompt = self._get_extraction_prompt(
                extracted_text,
                confirmed_section=confirmed_section
            )
            
            # 32k tokens to handle large POs (35+ items) without truncation
            raw_response = self._call_api(
                system_prompt, user_prompt, max_tokens=32768
            )
            
            data = self._parse_json(raw_response, context="po_extraction")
            
            # Handle possible nesting in model output
            if "purchaseOrder" in data:
                po_data = data["purchaseOrder"]
            else:
                po_data = data

            # Ensure items list exists
            if "items" not in po_data:
                po_data["items"] = []

            # Clean whitespace from all string fields before schema validation.
            po_data = self._clean_strings(po_data)

            # Validate with Pydantic
            po_obj = PurchaseOrder(**po_data)

            logger.info("Successfully parsed Purchase Order with NVIDIA Qwen API (single-pass)")
            return po_obj

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
            # 1. Strip systemic metadata tags like "T( Auto PO )", "C( Auto PO )", etc.
            # These are often artifacts from the source system or extraction labels
            # that we don't want in the final JSON.
            obj = re.sub(r'[A-Z]?\s*\(\s*Auto\s*PO\s*\)', '', obj, flags=re.IGNORECASE)

            # 2. Collapse all whitespace sequences (spaces, tabs, newlines) to one space
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

**CORE RULES:**
- Return ONLY valid JSON. No preamble, no markdown fences, no extra text.
- NEVER CALCULATE. NEVER FORMAT. Extract ONLY what is physically printed.
- Copy numeric values exactly as printed (e.g., "61,506" not 61506; "2,340.00" not 2340).
- Missing value → OMIT THE KEY ENTIRELY. Do NOT output `"field": null`. Just leave the key out to save tokens.
- Printed 0/0.00 → extract as-is `"0.00"`. Never omit or treat 0 as null.
- Do not derive, infer, or compute any field. Do not strip labels — extract values only.
- **LAYOUT ADAPTABILITY:** If standard headers/labels are missing or split across multiple lines, rely on semantic context and typical PO structures to map values to the correct schema fields.
- **SPEED OPTIMIZATION:** Return MINIFIED JSON without indentation, line breaks, or extra spaces.

---

### PURCHASE ORDER HEADER
- **purchaseOrder.poNumber** — PO Number / Purchase Order Number / P.O. Number / Buyer Order Number, exactly as printed.
- **purchaseOrder.poDate** — Purchase Order Date / PO Date / Date, exactly as printed.
- **purchaseOrder.expiryDate** — Expiry Date / P.O. Expiry / Valid Until, exactly as printed, else null.

---

### BUYER (entity issuing/receiving the PO)
**Identification:** Buyer is the company/store placing the order. In formal POs, name/address appear in header/letterhead. In retail-format POs, the store name printed prominently at top with address block is the buyer. Never use "Delivery Address" / "Ship To" as buyer address.
**PURGE TAGS:** Strictly remove system metadata tags like `T( Auto PO )`, `C( Auto PO )`, `B( Auto PO )` wherever they appear.

- **buyer.companyName** — Buyer/store/company name exactly as printed.
- **buyer.address** — Complete postal address (all lines, collected into one string). Extract the text lines immediately below the buyer company name (e.g., street, area, city) even if there is no "Address" label. Extract address part even if other fields (TIN, Phone) appear on the same line. Purge tags. Never use "Delivery Address"/"Ship To"/"Distribution Center" blocks.
- **buyer.gstno** — Only if label is exactly "GSTIN", "GSTN", or "GSTIN No". Not for "TIN"/"TIN No" labels. Value only. (Note: If buyer GSTNO is printed under the Delivery Address block, you MUST extract it here).
- **buyer.tin** — Only if label is exactly "TIN", "TIN No", or "Tax ID". Not for "GSTIN"/"GSTN" labels. Value only.
- **buyer.pan** — Only if label is exactly "PAN" or "PAN No". Format: 5 uppercase letters + 4 digits + 1 uppercase letter (e.g., CMWPM2648Q). Do NOT extract from within a GSTIN. Value only.
- **buyer.email** — Exactly as printed. (Note: If buyer Email is printed under the Delivery Address block, you MUST extract it here).
- **buyer.phone** — Exactly as printed.

---

### SELLER (vendor/supplier fulfilling the PO)
**Identification:** Seller is the vendor/supplier delivering goods, typically in a "Vendor"/"Seller"/"Supplier" section.

- **seller.companyName** — Exactly as printed.
- **seller.address** — Complete postal address, all lines collected. Purge tags.
- **seller.gstno** — Same label rules as buyer.gstno.
- **seller.tin** — Same label rules as buyer.tin.
- **seller.pan** — Same label rules as buyer.pan.
- **seller.email** — Exactly as printed, else null.
- **seller.phone** — Exactly as printed, else null.

---

### FINANCIAL SUMMARY
Extract only from the summary/footer/totals section. Do NOT sum item-level values. Do NOT calculate.

- **financialSummary.totalBasicValue** — Total Basic Value / Subtotal / Taxable Amount, else null.
- **financialSummary.totalCGST** — Total CGST amount, else null.
- **financialSummary.totalSGST** — Total SGST amount, else null.
- **financialSummary.totalIGST** — Total IGST amount, else null.
- **financialSummary.totalTaxAmt** — Combined Total GST / Total Tax Amount. Do NOT add CGST + SGST, else null.
- **financialSummary.totalOrderValue** — Grand total / Net Total / Total Order Value / Total Amount, exactly as printed.
- **financialSummary.discountPercent** — Overall discount %, else null.
- **financialSummary.discountAmt** — Overall discount amount, else null.
- **financialSummary.gstCompensationCess** — From summary footer exactly as printed, else null.
- **financialSummary.gstAdditionalCess** — From summary footer exactly as printed, else null.
- **financialSummary.totalQuantity** — Total Qty / Grand Total of Qty / Total Units, or the unlabeled number under the Qty column in the totals row (e.g., "NET TOTAL 174 3640.76" → 174). Do NOT confuse with grand total amount.

---

### LINE ITEMS
**General rules:**
- Extract each line item independently; do not merge separate items.
- A single item may span multiple text rows — collect all parts before mapping.
- Extract all values exactly as printed. Do not reformat, round, or calculate.
- **ZERO VALUES ARE DATA:** If a value is 0, 0.0, or 0.00, extract it exactly. Never treat 0 as null.

**Vertical Tax Stacking Rule (Reliance format):** One item occupies a block of 4 physical rows. Tax column headers are:
```
CGST(%) | CGST
SGST(%) | SGST
CESS(%) | CESS
CessFxdRt | CessFxdVl
```
Headers do NOT repeat per item. Map by row position within each item block.

Example — Item 1:
```
Row 1: 1  490001912  8906001050490  MOTHER RECP AP GINGER PCKLE 300G  1.000  CAR  3,150.00  2,340.00  2.50  58.50  2,340.00
Row 2:    20019000                  25.11.2025                                               2.50  58.50
Row 3:                              SAHO                                                     0.00   0.00
Row 4:                                                                 30.000  EA  105.00    0.00   0.00
```
Mandatory mapping: srNo=1, articleCode=490001912, eanCode=8906001050490, productDescription="MOTHER RECP AP GINGER PCKLE 300G", quantityCarton="1.000", uomCarton="CAR", mrpcarton="3,150.00", basicCostPrice="2,340.00", totalBaseValue="2,340.00", cgstPercent="2.50", cgstAmt="58.50" (Row 1), sgstPercent="2.50", sgstAmt="58.50" (Row 2), cessPercent="0.00", cessAmt="0.00" (Row 3), cessFxdRt="0.00", cessFxdVl="0.00" (Row 4 — **CRITICAL: must be "0.00", never null if a row exists**), quantityEach="30.000", uomEach="EA", mrpeach="105.00".

**Split-Row Item Assembly Rule:** Some PDFs render one item across two consecutive rows because the description overflows. Assemble as one item.srNo from Row A; articleCode and productDescription from Row B; all numerics from Row A.
**Detection signal:** A row has numeric price/qty/total but no articleCode, AND the next row has a numeric code + text but no numerics → same item.

---

**Field definitions for LINE ITEMS:**
- **srNo** — Serial/item/line number exactly as printed.
- **articleCode** — Article/item/product code exactly as printed.
- **hsnCode** — HSN/SAC code exactly as printed, else null.
- **eanCode** — EAN/barcode/UPC exactly as printed, else null.
- **vendorArticleNo** — Vendor article number if present as a distinct column, else null.
- **vendorItemNo** — Vendor item number if present as a distinct column, else null.
- **productDescription** — Full description exactly as printed, including wrapped text.
- **deliveryDate** — Item-level delivery date exactly as printed, else null.
- **quantityCarton** / **uomCarton** — Carton/box/case quantity and UOM if explicitly specified.
- **quantityEach** / **uomEach** — Each/unit quantity and UOM. If the PO has only one "Qty" or "Quantity" column without specifying carton vs each, map it here to quantityEach.
- **mrpcarton** — Carton/box MRP if explicitly specified.
- **mrpeach** — Unit MRP. If the PO has only one "MRP" column, map it here to mrpeach.
- **basicCostPrice** — Unit cost/rate BEFORE tax (Basic, Cost Price, Basic Rate).
- **landingRate** — Unit rate AFTER tax (Landing Rate, Landing, Rate incl. tax). Extract this even if the value is identical to the MRP.
- **gstPercent** — Combined GST % ONLY if a single "GST %" column exists.
- **cgstPercent** / **sgstPercent** / **igstPercent** — Component GST percentages if present.
- **cgstAmt** / **sgstAmt** / **igstAmt** / **gstAmt** — GST component amounts if present.
- **cessPercent** / **cessAmt** — CESS % and amount exactly as printed.
- **additionalCess** — GST Additional Cess at item level if present.
- **cessFxdVl** / **cessFxdRt** — CESS Fixed Value or Fixed Rate if present.
- **totalTaxAmt** — Total tax amount ONLY if explicitly labeled in item row.
- **totalBaseValue** — Taxable/base amount for the line item if present.
- **totalValueWithTax** — Final line item total inclusive of tax.
- **discountPercent** / **discountAmt** — Item-level discount % or amount if present.
"""
    
    def _get_extraction_prompt(
        self,
        extracted_text: str,
        confirmed_section: str = ""
    ) -> str:
        """Get the single comprehensive prompt for full PO extraction."""
        # Note: In a single pass, we provide the full schema structure.
        schema = {
            "purchaseOrder": {
                "poNumber": "string",
                "poDate": "string",
                "expiryDate": "string",
                "buyer": {
                    "companyName": "string",
                    "address": "string",
                    "gstno": "string",
                    "tin": "string",
                    "pan": "string",
                    "email": "string",
                    "phone": "string"
                },
                "seller": {
                    "companyName": "string",
                    "address": "string",
                    "gstno": "string",
                    "tin": "string",
                    "pan": "string",
                    "email": "string",
                    "phone": "string"
                },
                "financialSummary": {
                    "totalBasicValue": "string",
                    "totalCGST": "string",
                    "totalSGST": "string",
                    "totalIGST": "string",
                    "totalTaxAmt": "string",
                    "totalOrderValue": "string",
                    "discountPercent": "string",
                    "discountAmt": "string",
                    "gstCompensationCess": "string",
                    "gstAdditionalCess": "string",
                    "totalQuantity": "string"
                },
                "items": [
                    {
                        "srNo": "string",
                        "articleCode": "string",
                        "hsnCode": "string",
                        "eanCode": "string",
                        "vendorArticleNo": "string",
                        "vendorItemNo": "string",
                        "productDescription": "string",
                        "deliveryDate": "string",
                        "quantityCarton": "string",
                        "uomCarton": "string",
                        "quantityEach": "string",
                        "uomEach": "string",
                        "mrpcarton": "string",
                        "mrpeach": "string",
                        "basicCostPrice": "string",
                        "landingRate": "string",
                        "gstPercent": "string",
                        "cgstPercent": "string",
                        "sgstPercent": "string",
                        "igstPercent": "string",
                        "cgstAmt": "string",
                        "sgstAmt": "string",
                        "igstAmt": "string",
                        "gstAmt": "string",
                        "cessPercent": "string",
                        "cessAmt": "string",
                        "additionalCess": "string",
                        "cessFxdVl": "string",
                        "cessFxdRt": "string",
                        "totalTaxAmt": "string",
                        "totalBaseValue": "string",
                        "totalValueWithTax": "string",
                        "discountPercent": "string",
                        "discountAmt": "string"
                    }
                ]
            }
        }

        confirmed_block = f"\n\n{confirmed_section}\n" if confirmed_section else ""

        return f"""Read the following document text and fill in the JSON schema below.
The text layout uses spaces to preserve the original document structure — values on the same line may be side-by-side columns.

<DOCUMENT>
{extracted_text}
</DOCUMENT>{confirmed_block}

Fill this schema (return FULL purchaseOrder object):

<SCHEMA>
{json.dumps(schema, indent=2)}
</SCHEMA>

Return only valid MINIFIED JSON. Missing values = OMIT KEY."""
    # ── Legacy prompts (Deprecated) ───────────────────────────────────────────
    def _get_header_prompt(self, *args, **kwargs): return ""
    def _get_items_prompt(self, *args, **kwargs): return ""

