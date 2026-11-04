"""Two agents and a run id the project makes itself: created once, handed on, in the log line."""

import logging
import uuid

from agents import Agent, Runner

log = logging.getLogger("app")
writer = Agent(name="writer", instructions="Write.", model="gpt-4o-2024-08-06")
lead = Agent(name="lead", instructions="Lead.", handoffs=[writer], model="gpt-4o-2024-08-06")


def run(task: str) -> None:
    run_id = uuid.uuid4().hex
    log.info("start %s", run_id)
    Runner.run_sync(lead, task, context=dict(run_id=run_id))
