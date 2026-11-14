from google.adk.agents import LlmAgent
from pydantic import BaseModel


class Answer(BaseModel):
    text: str


def lookup(q: str) -> str:
    """Look something up."""
    return q


root_agent = LlmAgent(name="a", model="gemini-2.0-flash-001", tools=[lookup], output_schema=Answer)
