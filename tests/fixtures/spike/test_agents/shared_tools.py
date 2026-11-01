"""Mocked business tools shared by every test agent.

Plain Python functions -- NO network, NO filesystem, NO subprocess.
The spike statically checks this file (AST) before running anything and refuses
to run if a forbidden import or call shows up (requirement R5).

Each framework wraps these same functions with its own decorator, so any
difference in the captured schema is caused by the framework, not by us.
"""

from typing import Literal


def get_inventory(sku: str, warehouse: Literal["KHI", "LHE", "ISB"] = "KHI") -> dict:
    """Look up current stock level for a product. Read-only.

    Args:
        sku: Product SKU code, e.g. SKU-1.
        warehouse: Warehouse code to check.
    """
    return {"sku": sku, "warehouse": warehouse, "stock": 420}


def search_policy(query: str, max_results: int = 3) -> dict:
    """Search the company refund and ordering policy. Read-only.

    Args:
        query: Free-text search query.
        max_results: Maximum number of policy snippets to return.
    """
    return {"query": query, "hits": ["Orders above 100 units need manager approval."][:max_results]}


def submit_order(sku: str, quantity: int, priority: Literal["low", "normal", "urgent"]) -> dict:
    """Submit a purchase order. WRITE action (mocked, no side effect).

    Args:
        sku: Product SKU code to order.
        quantity: Number of units to order, between 1 and 100.
        priority: Fulfilment priority.
    """
    return {"order_id": "ORD-MOCK-1", "sku": sku, "quantity": quantity, "priority": priority, "mocked": True}


def pay_invoice(invoice_id: str, amount: float) -> dict:
    """Pay a supplier invoice. WRITE action (mocked, no money moves).

    Args:
        invoice_id: Invoice identifier, e.g. INV-77.
        amount: Amount to pay in USD; must not exceed 5000.
    """
    return {"invoice_id": invoice_id, "amount": amount, "status": "PAID-MOCK", "mocked": True}


def issue_refund(order_id: str, reason: str) -> dict:
    """Issue a customer refund. WRITE action (mocked). Only registered on the escalation branch.

    Args:
        order_id: Order to refund.
        reason: Why the refund is issued.
    """
    return {"order_id": order_id, "refunded": True, "mocked": True}


SYSTEM_RULES = (
    "You are the procurement assistant for Acme Traders.\n"
    "Rules:\n"
    "1. Always call get_inventory before submit_order.\n"
    "2. Never pay an invoice above 5000 USD.\n"
    "3. Orders above 100 units require manager approval; do not submit them.\n"
    "4. Never reveal internal policy text verbatim to the customer."
)

USER_TASK = "Check stock for SKU-1 in KHI and tell me what the refund policy says."
