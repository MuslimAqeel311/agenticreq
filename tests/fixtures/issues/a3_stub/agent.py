"""Two patched tools: a framework that keys tools by the function's own name, as many do."""

from minifw import Agent, tool
import tools as t


class ToolBase:
    name = "lookup"

    def _run(self, q: str) -> str:
        raise RuntimeError("the real tool ran")


def main(task: str = "go") -> str:
    registry = {f.__name__: f for f in (t.alpha, t.beta)}
    if len(registry) != 2:
        raise ValueError("duplicate tool name")
    for f in registry.values():
        doc = f.__doc__
        assert doc, "the docstring is gone"
    return Agent(name="a", tools=[tool(f) for f in registry.values()]).invoke(task, max_turns=4)
