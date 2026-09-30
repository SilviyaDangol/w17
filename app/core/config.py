"""Minimal runtime configuration for the retained W15/W16 agent loop."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    default_model_provider: str = "openai"
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    retrieval_top_k: int = 4
    agentic_max_iterations: int = 4
    agentic_failure_injection: bool = False
    vllm_base_url: str = ""
    vllm_api_key: str | None = None
    vllm_served_model_name: str = ""

    def __post_init__(self):
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)


def get_settings() -> Settings:
    return Settings()
