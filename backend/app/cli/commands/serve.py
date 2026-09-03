"""`orinth serve` — the one command allowed to import the server.

Every other command in this package is an HTTP client that imports `argparse`,
`json`, `tomllib`, and `httpx`. This one starts the backend, so importing
uvicorn and `app.main` is its entire job rather than a leak — and the imports
still live *inside* the handler, so `orinth --help` and `orinth serve --help`
pay nothing for them.

The directory flags set environment variables before that import, which is how
phase 18's desktop supervisor already relocates the data root: `Settings` reads
them at construction, and `app.main` constructs the container on import.
"""

import argparse
import os

from app.cli import output
from app.cli.commands._common import add_common
from app.cli.errors import EXIT_OK

ENV_FOR_FLAG = {
    "storage_dir": "STORAGE_DIR",
    "models_dir": "MODELS_DIR",
    "datasets_dir": "DATASETS_DIR",
    "database_url": "DATABASE_URL",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orinth serve", allow_abbrev=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--storage-dir", default=None)
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--datasets-dir", default=None)
    parser.add_argument("--database-url", default=None)
    add_common(parser)
    return parser


def run(argv: list[str], config, globals_) -> int:
    args = build_parser().parse_args(argv)

    for attribute, variable in ENV_FOR_FLAG.items():
        value = getattr(args, attribute, None)
        if value:
            os.environ[variable] = str(value)

    output.note(f"Starting Orinth on http://{args.host}:{args.port}")

    # The single sanctioned heavy import in this package, and it is deferred to
    # the moment the server is actually being started.
    import uvicorn

    uvicorn.run(
        "app.main:app" if args.reload else _app(),
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return EXIT_OK


def _app():
    from app.main import app

    return app
