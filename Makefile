.PHONY: backend frontend test lint

backend:
	cd backend && uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && pnpm dev

test:
	cd backend && uv run pytest

lint:
	cd frontend && pnpm lint
