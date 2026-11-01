"""Tools of the helpdesk agents: two read-only lookups and one write."""

import json

from helpdesk.minifw import tool


@tool
def lookup_order(order_id: str) -> str:
    """Return the status of an order."""
    return json.dumps({"order_id": order_id, "status": "shipped"})


@tool
def lookup_customer(email: str) -> str:
    """Return a customer's plan."""
    return json.dumps({"email": email, "plan": "pro"})


@tool
def send_refund(order_id: str, amount: float) -> str:
    """Send a refund to the customer's card."""
    raise RuntimeError("a real refund must never run in a mock run")


SUPPORT_TOOLS = [lookup_order, lookup_customer]
BILLING_TOOLS = [lookup_customer, send_refund]
