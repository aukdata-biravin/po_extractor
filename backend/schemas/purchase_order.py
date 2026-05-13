"""
Pydantic models for Purchase Order data structures.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class BuyerInfo(BaseModel):
    """Buyer/Procurement organization details."""
    companyName: Optional[str] = None
    address: Optional[str] = None
    gstno: Optional[str] = None
    tin: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class SellerInfo(BaseModel):
    """Seller/Vendor details."""
    companyName: Optional[str] = None
    address: Optional[str] = None
    email: Optional[str] = None
    gstno: Optional[str] = None
    tin: Optional[str] = None
    phone: Optional[str] = None



class FinancialSummary(BaseModel):
    """Financial summary with tax calculations."""
    totalBasicValue: Optional[str] = None
    totalCGST: Optional[str] = None
    totalSGST: Optional[str] = None
    totalIGST: Optional[str] = None
    totalTaxAmt: Optional[str] = None
    totalOrderValue: Optional[str] = None
    totalQuantity: Optional[str] = None
    discountPercent: Optional[str] = None
    discountAmt: Optional[str] = None
class OrderItem(BaseModel):
    """Individual line item in the purchase order."""
    srNo: Optional[str] = None
    articleCode: Optional[str] = None
    hsnCode: Optional[str] = None
    eanCode: Optional[str] = None
    productDescription: Optional[str] = None
    deliveryDate: Optional[str] = None
    quantityCarton: Optional[str] = None
    quantityEach: Optional[str] = None
    uomCarton: Optional[str] = None
    uomEach: Optional[str] = None
    mrpcarton: Optional[str] = None
    mrpeach: Optional[str] = None
    basicCostPrice: Optional[str] = None
    landingRate: Optional[str] = None
    gstPercent: Optional[str] = None
    cgstPercent: Optional[str] = None
    sgstPercent: Optional[str] = None
    igstPercent: Optional[str] = None
    cgstAmt: Optional[str] = None
    sgstAmt: Optional[str] = None
    igstAmt: Optional[str] = None
    gstAmt: Optional[str] = None
    cessPercent: Optional[str] = None
    cessAmt: Optional[str] = None
    cessFxdVl: Optional[str] = None
    cessFxdRt: Optional[str] = None
    totalTaxAmt: Optional[str] = None
    totalBaseValue: Optional[str] = None
    totalValueWithTax: Optional[str] = None
    discountPercent: Optional[str] = None
    discountAmt: Optional[str] = None


class PurchaseOrder(BaseModel):
    """Complete Purchase Order structure."""
    poNumber: Optional[str] = None
    poDate: Optional[str] = None
    expiryDate: Optional[str] = None
    buyer: Optional[BuyerInfo] = None
    seller: Optional[SellerInfo] = None
    financialSummary: Optional[FinancialSummary] = None
    items: Optional[List[OrderItem]] = None


class PurchaseOrderResponse(BaseModel):
    """API response containing extracted Purchase Order."""
    purchaseOrder: PurchaseOrder
