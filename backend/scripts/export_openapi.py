"""Export the FastAPI app's OpenAPI schema to a JSON file.

This builds the schema in-process from the `FastAPI` app object
(`app.openapi()`) and never binds a socket or starts uvicorn, so it is safe to
run in CI or as a local pre-codegen step without a running backend.

Usage:

    cd backend
    uv run python scripts/export_openapi.py [output_path]

`output_path` defaults to `backend/openapi.json` when omitted. The frontend's
`pnpm generate:api` script (see `frontend/package.json`) invokes this to
produce the JSON schema consumed by `openapi-typescript`, which regenerates
`frontend/types/generated/api.ts`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402  (import after sys.path setup)


def _mark_optional_fields_as_always_present(schema: dict[str, Any]) -> None:
    """Work around a FastAPI/Pydantic quirk that makes generated types too optional.

    FastAPI's OpenAPI export omits the JSON Schema `default` keyword for any
    field whose Python default is `None`, and never surfaces
    `Field(default_factory=...)` values at all. `openapi-typescript` treats a
    property as TypeScript-optional (`field?:`) unless it is in the schema's
    `required` list *or* carries an explicit `default` key -- so, without this
    fix-up, every `Optional[...] = None` and `default_factory` field would be
    typed as possibly-`undefined`, even though Pydantic always serializes
    every declared field (using `null`/`[]`/`{}` for "unset" ones, never
    omitting the key). That mismatch would force needless `?? []`/optional
    chaining throughout the frontend and make response types weaker than the
    real API contract.

    This walks every component schema and injects `"default": null` on any
    property that is not already `required` and does not already carry a
    `default`, so `openapi-typescript` generates it as a required (but still
    possibly-null/empty) field. This does not change validation behavior --
    it only edits the *exported* schema copy used for type generation.
    """
    for schema_obj in schema.get("components", {}).get("schemas", {}).values():
        if not isinstance(schema_obj, dict):
            continue
        properties = schema_obj.get("properties")
        if not isinstance(properties, dict):
            continue
        required = set(schema_obj.get("required", []))
        for name, prop in properties.items():
            if name in required or not isinstance(prop, dict):
                continue
            prop.setdefault("default", None)


def export_openapi(output_path: Path) -> None:
    """Write `app.openapi()` as pretty-printed, sorted JSON to `output_path`."""
    schema = app.openapi()
    _mark_optional_fields_as_always_present(schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    output_path = Path(argv[1]) if len(argv) > 1 else BACKEND_ROOT / "openapi.json"
    export_openapi(output_path)
    print(f"Wrote OpenAPI schema to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
