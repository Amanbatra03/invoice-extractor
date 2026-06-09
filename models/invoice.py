from pydantic import BaseModel


class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    total: float | None = None


class InvoiceSchema(BaseModel):
    vendor_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total_amount: float | None = None
    currency: str | None = None
    line_items: list[LineItem] = []
