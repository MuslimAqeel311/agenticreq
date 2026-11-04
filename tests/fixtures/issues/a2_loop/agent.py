"""A library retries a refused temp-file write in a loop, the way tempfile does on Windows."""

import os
import time

from minifw import Agent, tool


def load_config() -> None:
    attempts = 0
    for i in range(10_000):
        attempts += 1
        try:
            os.close(os.open(f"state-{i}.tmp", os.O_CREAT | os.O_WRONLY))
            return
        except PermissionError:
            continue
    time.sleep(600)


@tool
def lookup(q: str) -> str:
    """Look up."""
    return "x"


def main(task: str = "go") -> str:
    load_config()
    return Agent(name="a", tools=[lookup]).invoke(task, max_turns=3)
