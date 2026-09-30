import asyncio
import json
from types import SimpleNamespace

from app.core.config import Settings
from app.schemas.chat import AgenticResearchRequest
from app.services.agentic_verification import AgenticVerificationLoop, RetrievedChunk


class FakeVectorStore:
    def search(self, query, limit):
        return [RetrievedChunk("doc", "fixture.md", query, "verified evidence", 0.1)]


def response(body):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_agentic_loop_runs_multiple_iterations():
    calls = 0

    async def complete(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return response({"action": "search_documents", "query": "policy", "answer": None, "clarification_question": None, "self_check": "Need evidence."})
        return response({"action": "finish", "query": None, "answer": "Verified.", "clarification_question": None, "self_check": "Supported."})

    result = asyncio.run(AgenticVerificationLoop(Settings(openai_api_key="test"), complete, FakeVectorStore()).run(AgenticResearchRequest(query="q", provider="openai")))
    assert result.outcome == "finish"
    assert result.iterations == 2
    assert result.tool_calls == 1


def test_agentic_loop_stops_safely_at_max_iterations():
    async def complete(**kwargs):
        return response({"action": "search_documents", "query": "more", "answer": None, "clarification_question": None, "self_check": "Need more."})

    result = asyncio.run(AgenticVerificationLoop(Settings(agentic_max_iterations=2, openai_api_key="test"), complete, FakeVectorStore()).run(AgenticResearchRequest(query="q", provider="openai")))
    assert result.outcome == "insufficient_evidence"
    assert result.iterations == 2
    assert result.answer.startswith("Insufficient evidence")


def test_malformed_retrieval_never_becomes_a_confident_answer():
    async def complete(**kwargs):
        return response({"action": "finish", "query": None, "answer": "unsupported", "clarification_question": None, "self_check": "Supported."})

    result = asyncio.run(AgenticVerificationLoop(Settings(agentic_failure_injection=True, openai_api_key="test"), complete, FakeVectorStore()).run(AgenticResearchRequest(query="q", provider="openai")))
    assert result.outcome == "insufficient_evidence"
    assert result.sources == []
