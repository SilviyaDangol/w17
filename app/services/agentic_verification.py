"""Bounded single-agent self-checking research loop for the W16 assignment."""

import json
import logging
import math
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.schemas.chat import (
    AgenticDecision,
    AgenticResearchRequest,
    AgenticResearchResponse,
    AgenticTraceStep,
    AgenticTokenUsage,
    Source,
)

logger = logging.getLogger(__name__)


async def acompletion(**kwargs: Any) -> Any:
    """Small OpenAI-compatible client; avoids a heavyweight client dependency."""
    import asyncio
    return await asyncio.to_thread(_completion_sync, kwargs)


def _completion_sync(kwargs: dict[str, Any]) -> Any:
    model = kwargs["model"].removeprefix("openai/")
    body = {key: value for key, value in kwargs.items() if key in {"messages", "temperature", "top_p", "response_format"}}
    body["model"] = model
    request = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {kwargs['api_key']}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    from types import SimpleNamespace
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload["choices"][0]["message"]["content"]))], usage=SimpleNamespace(**payload.get("usage", {})))


@dataclass
class RetrievedChunk:
    """Minimal W15 retrieval contract used by the bounded W16 agent."""

    document_id: str
    filename: str
    chunk_id: str
    text: str
    distance: float

    def as_source(self) -> Source:
        return Source(document_id=self.document_id, filename=self.filename, chunk_id=self.chunk_id, excerpt=self.text)

AGENTIC_SYSTEM_PROMPT = """You are a document research agent that must verify its own answer.
Return exactly one JSON decision with these keys:
action, query, answer, clarification_question, self_check.
Before emitting a decision, privately draft the answer from the current evidence
and check whether every claim in that draft is supported by the evidence notes.
Choose search_documents when the current evidence does not support every claim.
Choose ask_clarification when the user's request is ambiguous. Choose finish
only when the evidence supports the answer. Never invent source metadata.
"""


class AgenticVerificationLoop:
    """Runs one model decision and at most one retrieval tool call per iteration."""

    def __init__(
        self,
        settings: Settings | None = None,
        completion_fn: Callable[..., Awaitable[Any]] | None = None,
        vector_store: Any | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.completion_fn = completion_fn or acompletion
        self.vector_store = vector_store
        self.system_prompt = system_prompt or AGENTIC_SYSTEM_PROMPT

    async def run(self, request: AgenticResearchRequest) -> AgenticResearchResponse:
        provider = request.provider or self.settings.default_model_provider
        scratchpad: dict[str, list[str]] = {
            "queries_tried": [],
            "evidence_summaries": [],
            "self_check_verdicts": [],
        }
        sources: dict[str, Source] = {}
        token_usage = AgenticTokenUsage()
        tool_calls = 0
        self_checks: list[str] = []
        trace: list[AgenticTraceStep] = []
        outcome = "insufficient_evidence"
        answer = "Insufficient evidence: unable to fully verify an answer within the iteration limit."
        clarification_question: str | None = None

        for iteration in range(1, self.settings.agentic_max_iterations + 1):
            response = await self._complete(provider, request.query, scratchpad)
            token_usage = _add_usage(token_usage, _usage(response))
            decision = _parse_decision(response)
            self_checks.append(decision.self_check)
            scratchpad["self_check_verdicts"].append(decision.self_check)

            if decision.action == "finish":
                # A finish with no validated evidence is never allowed to become
                # a confident answer based on the model's unsupported claims.
                if sources:
                    outcome, answer = "finish", decision.answer or ""
                    trace.append(AgenticTraceStep(step=iteration, action="finish", reasoning=decision.self_check, result="success"))
                break
            if decision.action == "ask_clarification":
                outcome = "ask_clarification"
                clarification_question = decision.clarification_question
                answer = clarification_question or "Please clarify the question."
                trace.append(AgenticTraceStep(step=iteration, action="ask_clarification", reasoning=decision.self_check, result="clarification requested"))
                break

            query = decision.query or ""
            scratchpad["queries_tried"].append(query)
            tool_calls += 1
            raw_chunks = self._retrieve(query)
            chunks = _sanitize_chunks(raw_chunks)
            raw_result = None if chunks is None else [
                {"document_id": chunk.document_id, "filename": chunk.filename, "chunk_id": chunk.chunk_id, "text": chunk.text, "distance": chunk.distance}
                for chunk in chunks
            ]
            trace.append(AgenticTraceStep(step=iteration, action="search_documents", reasoning=decision.self_check, tool="search_documents", arguments={"query": query, "limit": min(self.settings.retrieval_top_k, 5)}, result=json.dumps(raw_result)))
            if chunks is None:
                logger.warning("agentic_malformed_retrieval query=%r", query)
                scratchpad["evidence_summaries"].append("Search failed validation; no evidence was added.")
                continue
            for chunk in chunks:
                sources[chunk.chunk_id] = chunk.as_source()
                scratchpad["evidence_summaries"].append(
                    f"{chunk.chunk_id} ({chunk.filename}): {chunk.text[:240].strip()}"
                )

        return AgenticResearchResponse(
            answer=answer,
            sources=list(sources.values()),
            outcome=outcome,
            clarification_question=clarification_question,
            iterations=iteration,
            tool_calls=tool_calls,
            queries_tried=scratchpad["queries_tried"],
            self_checks=self_checks,
            token_usage=token_usage,
            trace=trace,
            termination_reason=("success" if outcome == "finish" else "clarification" if outcome == "ask_clarification" else "max_iterations_or_unverified"),
        )

    def _retrieve(self, query: str) -> Any:
        if self.settings.agentic_failure_injection:
            return [{"document_id": "truncated", "text": "malformed result"}]
        if self.vector_store is None:
            raise RuntimeError("A retrieval implementation must be supplied to the agent loop.")
        return self.vector_store.search(query, min(self.settings.retrieval_top_k, 5))

    async def _complete(self, provider: str, query: str, scratchpad: dict[str, list[str]]) -> Any:
        request: dict[str, Any] = {
            "messages": [
                {"role": "developer", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"User question: {query}\nCompact scratchpad:\n{json.dumps(scratchpad)}",
                },
            ],
            "temperature": self.settings.llm_temperature,
            "top_p": self.settings.llm_top_p,
            "response_format": {"type": "json_object"},
        }
        return await self.completion_fn(**self._provider_options(provider), **request)

    def _provider_options(self, provider: str) -> dict[str, str]:
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for the OpenAI route.")
            return {"model": f"openai/{self.settings.openai_model}", "api_key": self.settings.openai_api_key}
        if provider == "vllm":
            return {
                "model": f"openai/{self.settings.vllm_served_model_name}",
                "api_base": self.settings.vllm_base_url,
                "api_key": self.settings.vllm_api_key or "local-vllm",
            }
        raise ValueError("Provider must be either 'openai' or 'vllm'.")


def _parse_decision(response: Any) -> AgenticDecision:
    try:
        content = response.choices[0].message.content
        payload = json.loads(content)
        if not isinstance(payload.get("self_check"), str) or not payload["self_check"].strip():
            payload["self_check"] = "No self-check rationale was supplied by the model."
        return AgenticDecision.model_validate(payload)
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as error:
        raise ValueError("Model returned an invalid agentic decision.") from error


def _usage(response: Any) -> AgenticTokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return AgenticTokenUsage()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", prompt + completion) or 0)
    return AgenticTokenUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _add_usage(left: AgenticTokenUsage, right: AgenticTokenUsage) -> AgenticTokenUsage:
    return AgenticTokenUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


def _sanitize_chunks(raw: Any) -> list[RetrievedChunk] | None:
    """Reject the complete tool result if any record is malformed or corrupt."""
    if not isinstance(raw, list) or not raw:
        return None
    clean: list[RetrievedChunk] = []
    for item in raw:
        if not isinstance(item, RetrievedChunk):
            return None
        if not all(isinstance(getattr(item, field), str) and getattr(item, field).strip() for field in ("document_id", "filename", "chunk_id", "text")):
            return None
        if not isinstance(item.distance, (int, float)) or not math.isfinite(item.distance):
            return None
        clean.append(item)
    return clean
