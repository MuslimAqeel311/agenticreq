from dataclasses import dataclass


@dataclass
class Settings:
    model: str = "gpt-4o-2024-08-06"


settings = Settings()
MODEL_NAME = "claude-sonnet-4-5-20250929"
