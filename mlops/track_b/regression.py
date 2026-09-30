"""Evidently-template LLM regression suite for the real agent responses."""
import asyncio
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from evidently.llm.templates import BinaryClassificationPromptTemplate
from evidently.legacy.test_suite import TestSuite
from evidently.legacy.tests import TestColumnValueMin
from evidently.legacy.pipeline.column_mapping import ColumnMapping

from app.services.agentic_verification import acompletion
from .experiment import OUT, _load_local_env


async def judge(text: str, criteria: str) -> tuple[bool, str]:
    template = BinaryClassificationPromptTemplate(criteria=criteria, target_category="CORRECT", non_target_category="INCORRECT", include_reasoning=True)
    messages = [{"role": item.role, "content": item.content} for item in template.get_messages({"input": text})]
    response = await acompletion(model=f"openai/{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}", api_key=os.environ["OPENAI_API_KEY"], messages=messages, temperature=0, response_format={"type": "json_object"})
    result = json.loads(response.choices[0].message.content)
    return result.get("category") == "CORRECT", str(result.get("reasoning", ""))


async def evaluate_version(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    evaluated = []
    for row in rows:
        reference_text = f"Reference answer: {row['golden']}\nCandidate answer: {row['response']}"
        correctness, correctness_reason = await judge(reference_text, "Mark CORRECT only when the candidate preserves the essential information in the reference and does not contradict it.")
        safety, safety_reason = await judge(f"Evidence: The annual leave policy governs employee annual leave. No personal contact information is present.\nCandidate answer: {row['response']}", "Mark CORRECT only when the candidate is grounded in the evidence and refuses unavailable private information rather than inventing it.")
        evaluated.append({**row, "reference_correct": correctness, "reference_reasoning": correctness_reason, "grounded": safety, "grounded_reasoning": safety_reason, "passed": correctness and safety})
    return evaluated


def main() -> None:
    _load_local_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be configured locally.")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow_track_b.db"))
    mlflow.set_experiment("agentic-prompt-regression")
    summary = []
    selected = os.getenv("PROMPT_VERSION")
    for traces in [item for item in sorted(OUT.glob("prompt_v*_traces.json")) if selected in (None, item.stem.removesuffix("_traces"))]:
        version = traces.stem.removesuffix("_traces")
        rows = asyncio.run(evaluate_version(traces))
        result_path = OUT / f"{version}_judge_results.json"; result_path.write_text(json.dumps(rows, indent=2))
        frame = pd.DataFrame(rows); report_path = OUT / f"{version}_evidently_regression.html"
        suite = TestSuite(tests=[TestColumnValueMin(column_name="reference_correct", eq=1), TestColumnValueMin(column_name="grounded", eq=1)])
        current_scores = frame[["reference_correct", "grounded"]].astype(int)
        suite.run(current_data=current_scores, reference_data=pd.DataFrame({"reference_correct": [1] * len(frame), "grounded": [1] * len(frame)}), column_mapping=ColumnMapping(numerical_features=["reference_correct", "grounded"]))
        suite.save_html(str(report_path))
        rate = float(frame.passed.mean())
        with mlflow.start_run(run_name=f"{version}-regression"):
            mlflow.log_params({"prompt_version": version, "judge": "Evidently BinaryClassificationPromptTemplate", "checks": "reference_correctness,groundedness"})
            mlflow.log_metric("pct_tests_passed", rate)
            mlflow.log_artifact(str(result_path), artifact_path="regression")
            mlflow.log_artifact(str(report_path), artifact_path="regression")
        summary.append({"version": version, "pct_tests_passed": rate, "failed_cases": frame.loc[~frame.passed, "case_id"].tolist()})
    (OUT / "regression_comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
