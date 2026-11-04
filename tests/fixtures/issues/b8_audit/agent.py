"""An audit trail through a logger, not a variable called audit_log."""

import logging
from pathlib import Path

from pydantic_ai import Agent

audit_trail = logging.getLogger("audit.payments")
log = logging.getLogger("app")



def pay(ref: str) -> str:
    """Pay."""
    audit_trail.info("pay %s", ref)
    Path("ledger.txt").write_text(ref)
    return "ok"


agent = Agent("openai:gpt-4o-2024-08-06", tools=[pay])
