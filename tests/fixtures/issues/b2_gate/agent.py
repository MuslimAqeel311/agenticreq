"""Approval by behaviour: the gate function has no approval word the old table knew."""

from pathlib import Path

from agents import Agent, function_tool

APPROVED = {"lead"}


def _approved(who: str) -> bool:
    return who in APPROVED


@function_tool
def pay_invoice(who: str, ref: str) -> str:
    """Pay an invoice."""
    if not _approved(who):
        return "refused"
    Path("ledger.txt").write_text(ref)
    return "paid"


@function_tool
def refund(ref: str) -> str:
    """Refund a payment. The check comes after the write, so it gates nothing."""
    Path("ledger.txt").write_text(ref)
    if not _approved(ref):
        return "refused"
    return "done"


@function_tool
def archive(ref: str) -> str:
    """Archive a record. A plain empty check is not a gate."""
    if not ref:
        return "nothing"
    Path("ledger.txt").write_text(ref)
    return "done"


agent = Agent(name="payer", instructions="Pay.", tools=[pay_invoice, refund, archive], model="gpt-4o-2024-08-06")
