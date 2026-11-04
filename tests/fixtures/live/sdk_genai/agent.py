"""Plain google-genai SDK with automatic function calling."""
from google import genai
from google.genai import types


def lookup(q: str) -> str:
    """Look something up."""
    return "hit"


def main(task: str = "go") -> str:
    client = genai.Client(api_key="x")
    cfg = types.GenerateContentConfig(system_instruction="Help.", tools=[lookup])
    return client.models.generate_content(model="gemini-2.0-flash-001", contents=task, config=cfg).text or ""
