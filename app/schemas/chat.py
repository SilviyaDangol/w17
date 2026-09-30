from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What does the literature review recommend for resume parsing?",
                    "provider": "openai",
                },
                {
                    "query": "What are the main TA-MAS research themes?",
                    "provider": "vllm",
                },
                {
                    "query": "What documents have been indexed?"
                },
            ]
        }
    )

    query: str = Field(min_length=1, max_length=4_000, description="The user's question.")
    provider: Literal["openai", "vllm"] | None = Field(
        default=None,
        description="Model route: 'openai' for the hosted OpenAI model, 'vllm' for your local/Colab vLLM server. Omit to use DEFAULT_MODEL_PROVIDER.",
        examples=["openai", "vllm"],
    )
    allow_fallback: bool = Field(
        default=False,
        description="Only for vLLM: retry vLLM first, then use OpenAI if vLLM remains unavailable.",
    )


class Source(BaseModel):
    """A retrieved document citation. Populated in the RAG milestone."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    chunk_id: str
    excerpt: str


class SearchDocumentsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    limit: int = Field(ge=1, le=5)


class ToolUse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: SearchDocumentsArguments
    result: str


class GeneratedAnswer(BaseModel):
    """The model-generated portion of a response, before delivery metadata."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[Source]
    tool_uses: list[ToolUse]


class ResponseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_provider: Literal["openai", "vllm"] = "openai"
    actual_provider: Literal["openai", "vllm"] = "openai"
    cache_hit: bool = False
    fallback_used: bool = False
    latency_ms: float = Field(default=0.0, ge=0)


class AssistantAnswer(GeneratedAnswer):
    """The stable public JSON contract returned by every chat endpoint."""

    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


class BatchChatRequest(BaseModel):
    requests: list[ChatRequest] = Field(min_length=1, max_length=10)


class BatchChatItem(BaseModel):
    index: int
    answer: AssistantAnswer | None = None
    error: str | None = None


class BatchChatResponse(BaseModel):
    results: list[BatchChatItem]


class AgenticResearchRequest(ChatRequest):
    """Request for the W16 self-checking research loop."""


class AgenticDecision(BaseModel):
    """One model decision; the application validates it before any action."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["search_documents", "ask_clarification", "finish"]
    query: str | None = None
    answer: str | None = None
    clarification_question: str | None = None
    self_check: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "AgenticDecision":
        required = {
            "search_documents": (self.query, "query"),
            "ask_clarification": (self.clarification_question, "clarification_question"),
            "finish": (self.answer, "answer"),
        }
        value, name = required[self.action]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required for action={self.action}")
        return self


class AgenticTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AgenticTraceStep(BaseModel):
    """Auditable agent decision or tool event for MLOps evaluation."""

    step: int
    action: str
    reasoning: str
    tool: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)
    result: str | None = None


class AgenticResearchResponse(BaseModel):
    answer: str
    sources: list[Source]
    outcome: Literal["finish", "ask_clarification", "insufficient_evidence"]
    clarification_question: str | None = None
    iterations: int = Field(ge=1)
    tool_calls: int = Field(ge=0)
    queries_tried: list[str]
    self_checks: list[str]
    token_usage: AgenticTokenUsage
    trace: list[AgenticTraceStep] = Field(default_factory=list)
    termination_reason: str = "max_iterations"
