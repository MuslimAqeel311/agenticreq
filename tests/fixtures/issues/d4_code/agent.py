"""A CodeAgent: tools reach the model as prompt text, never as wire schemas."""

from smolagents import CodeAgent, LiteLLMModel, tool


@tool
def lookup(q: str) -> str:
    """Look up.

    Args:
        q: the query
    """
    return q


@tool
def save_note(text: str) -> str:
    """Save a note.

    Args:
        text: the note
    """
    open("notes.txt", "a").write(text)
    return "ok"


agent = CodeAgent(tools=[lookup, save_note], model=LiteLLMModel(model_id="gpt-4o-2024-08-06"), max_steps=4)


def main(task: str = "go") -> str:
    return str(agent.run(task))
