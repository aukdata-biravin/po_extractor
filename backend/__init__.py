"""
PO Extraction — backend package
================================
Provides the complete workflow for extracting structured Purchase Order data
from PDF documents: OCR detection → text extraction → AI parsing.

Typical usage (as a library)
-----------------------------
    from extractor import POExtractor

    extractor = POExtractor()
    result = extractor.extract_from_pdf("path/to/po.pdf")
    print(result.purchaseOrder.poNumber)
"""

# Absolute imports work whether the backend folder is used as a flat module
# directory (Docker / uvicorn) or imported as a Python package.
from extractor import POExtractor
from schemas import (
    PurchaseOrder,
    PurchaseOrderResponse,
    BuyerInfo,
    SellerInfo,
    FinancialSummary,
    OrderItem,
)

__all__ = [
    "POExtractor",
    "PurchaseOrder",
    "PurchaseOrderResponse",
    "BuyerInfo",
    "SellerInfo",
    "FinancialSummary",
    "OrderItem",
]
