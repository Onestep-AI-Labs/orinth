"""What the kernel can see about how the workspace is configured.

`provider_config()` never returns the OpenRouter key. That is a rule about the
SDK's shape, not a security boundary — a kernel runs arbitrary code and can read
`backend/.env` directly, so pretending otherwise would be a false assurance. The
reason to omit it is narrower and still worth honouring: a key that is never
returned is a key that never lands in a notebook's saved output and gets
committed by accident.
"""

from orinth import _workspace


def workspace() -> dict:
    resolved = _workspace.settings()
    storage = _workspace.storage()
    return {
        "storage": str(storage.root),
        "datasets": str(storage.datasets),
        "models": str(resolved.models_path),
        "notebooks": str(storage.notebooks),
        "api_base": _workspace.api_base(),
        "notebook_id": _workspace.notebook_id(),
        "project_id": _workspace.default_project_id(),
    }


def provider_config() -> dict:
    resolved = _workspace.settings()
    return {
        "openrouter_model": resolved.openrouter_model,
        "openrouter_key_configured": bool(resolved.openrouter_key),
    }
