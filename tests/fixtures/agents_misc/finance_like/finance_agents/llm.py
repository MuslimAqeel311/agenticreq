import os

from langchain_anthropic import ChatAnthropic


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(model=os.getenv("FINANCE_MODEL", "claude-sonnet-5-5"), max_tokens=1024)
