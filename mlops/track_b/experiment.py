"""Run real prompt versions through the W16 agent loop and log MLflow traces."""
import asyncio
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from app.core.config import Settings
from app.schemas.chat import AgenticResearchRequest
from app.services.agentic_verification import AGENTIC_SYSTEM_PROMPT, AgenticVerificationLoop, RetrievedChunk

OUT = Path("artifacts/track_b")
PROMPTS = sorted(Path("prompts").glob("prompt_v*.txt"))
CASES = [
    {"id": "leave", "query": "How is annual leave handled?", "golden": "Annual leave is governed by the policy."},
    {"id": "source", "query": "What source supports the leave answer?", "golden": "The annual leave policy is the source."},
    {"id": "unknown", "query": "What is the CEO's private phone number?", "golden": "Insufficient evidence is available in the documents."},
]


class FixtureStore:
    def search(self, query: str, limit: int):
        return [RetrievedChunk("policy-1", "annual_leave_policy.md", "leave-1", "The annual leave policy governs employee annual leave. No personal contact information is present.", 0.1)]


async def _run_version(name: str, revision: str):
    key = os.environ["OPENAI_API_KEY"]
    settings = Settings(openai_api_key=key, openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), llm_temperature=0, agentic_max_iterations=4)
    loop = AgenticVerificationLoop(settings=settings, vector_store=FixtureStore(), system_prompt=f"{AGENTIC_SYSTEM_PROMPT}\n\nPrompt revision ({name}):\n{revision}\nAlways search before finish.")
    records = []
    for case in CASES:
        try:
            result = await loop.run(AgenticResearchRequest(query=case["query"], provider="openai"))
            records.append({"case_id": case["id"], "query": case["query"], "golden": case["golden"], "response": result.answer, "outcome": result.outcome, "trace": [step.model_dump() for step in result.trace], "iterations": result.iterations, "termination_reason": result.termination_reason, "tokens": result.token_usage.model_dump()})
        except Exception as error:
            records.append({"case_id": case["id"], "query": case["query"], "golden": case["golden"], "response": "", "outcome": "hard_failure", "trace": [{"step": 1, "action": "parse_error", "reasoning": str(error), "tool": None, "arguments": {}, "result": None}], "iterations": 1, "termination_reason": "invalid_model_decision", "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
    return records


def main() -> None:
    _load_local_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set locally; never commit it.")
    OUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_track_b.db"))
    mlflow.set_experiment("agentic-prompt-regression")
    selected = os.getenv("PROMPT_VERSION")
    comparison = []
    for path in [item for item in PROMPTS if selected in (None, item.stem)]:
        name, revision = path.stem, path.read_text()
        records = asyncio.run(_run_version(name, revision))
        trace_path = OUT / f"{name}_traces.json"; trace_path.write_text(json.dumps(records, indent=2))
        completed = sum(item["outcome"] in {"finish", "insufficient_evidence"} for item in records)
        avg_iterations = sum(item["iterations"] for item in records) / len(records)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"prompt_version": name, "temperature": 0, "retrieval_top_k": 4, "max_iterations": 4})
            mlflow.log_metrics({"task_completion_rate": completed / len(records), "avg_iterations": avg_iterations, "total_tokens": sum(item["tokens"]["total_tokens"] for item in records)})
            mlflow.log_text(revision, f"prompts/{path.name}")
            mlflow.log_artifact(str(trace_path), artifact_path="traces")
        comparison.append({"prompt_version": name, "task_completion_rate": completed / len(records), "avg_iterations": avg_iterations, "total_tokens": sum(item["tokens"]["total_tokens"] for item in records)})
        print(f"{name}: {completed}/{len(records)} completed; traces={trace_path}")
    comparison_path = OUT / "prompt_run_comparison.csv"
    pd.DataFrame(comparison).to_csv(comparison_path, index=False)


def _load_local_env() -> None:
    """Load local key settings without printing or committing them."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


if __name__ == "__main__":
    main()
