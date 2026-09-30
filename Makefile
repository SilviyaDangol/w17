setup:
	uv sync --extra dev

train:
	uv run python -m mlops.track_a.train

monitor:
	uv run python -m mlops.track_a.monitor

serve:
	uv run uvicorn mlops.track_a.serve:app --port 8001

test:
	uv run --extra dev pytest
