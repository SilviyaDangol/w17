# Week 17 MLOps — Track B: Agentic AI

This branch contains only the W15/W16 source required for the Week 17 agentic MLOps deliverables: the bounded research loop, its schemas/configuration, versioned prompts, experiments, traces, regression evaluation, reports, and lockfile. Earlier UI, Docker, notebook, and unrelated source files are intentionally excluded.

## Reproduce

Create a local `.env` (never commit it) containing `OPENAI_API_KEY` and optionally `OPENAI_MODEL`. Then run:

```bash
uv sync --extra dev
uv run python -m mlops.track_b.experiment
uv run python -m mlops.track_b.regression
```

`uv.lock` makes the environment reproducible from a clean clone. The project uses a small OpenAI-compatible client instead of a heavyweight SDK, while still making real OpenAI API calls.

## Tracking strategy

`prompts/prompt_v1.txt`, `prompt_v2.txt`, and `prompt_v3.txt` are explicit prompt versions. Each is run through the actual W16 bounded agent loop on three fixed queries. MLflow records prompt version/configuration, task-completion rate, iterations, token usage, full JSON traces, and regression pass rate.

The traces showed v1 repeatedly searched after receiving evidence and asked needless clarifying questions. v2 improved completion to 2/3 but still failed the regression checks. v3 explicitly responds to the remaining failure: after one valid search it finishes with the supported answer/source, and it refuses unavailable private information without clarification. The real runs improved completion from 1/3 (v1), to 2/3 (v2), to 3/3 (v3).

## Regression suite and monitoring

The golden answers are stored alongside each fixed query in `mlops/track_b/experiment.py`. `regression.py` uses Evidently's `BinaryClassificationPromptTemplate` with a real LLM judge for two checks per response:

- Reference correctness: preserves the golden answer without contradiction.
- Groundedness/safety: is supported by the available evidence and refuses unavailable private information.

The resulting judge JSON, run-comparison CSV, sanity check, and Evidently HTML reports are under `artifacts/track_b/`. `pct_tests_passed` is logged to MLflow for every configuration. A version with failed tests is not promoted. In the committed run, v1/v2 scored 0%; after the final trace-driven revision, v3 scored 100% (3/3) and is the promotion candidate. `judge_sanity_check.md` compares the judge verdicts against manual trace/response review.

Run the local MLflow UI with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow_track_b.db --port 5000
```

Airflow is optional bonus work and is intentionally not included.
