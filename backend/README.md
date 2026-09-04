# Backend

FastAPI backend for model registry, inference, evaluation jobs, and training jobs.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The ML dependencies are loaded lazily when a model is used. API docs are available at `/docs`.

## Database migrations

Schema changes are managed with [Alembic](https://alembic.sqlalchemy.org/). On
startup, `app.core.database.init_db()` brings the database up to date
automatically:

- A brand-new database (no file yet) is migrated straight to `head`, which
  creates the full schema.
- A database created by an older, pre-Alembic version of this app (tables
  exist but there is no `alembic_version` table) is stamped at the baseline
  revision — its schema already matches it — and then upgraded to `head` in
  case newer migrations have landed since.
- A database that already has an `alembic_version` table is simply upgraded
  to `head`.

You do not need to run anything manually for local development; starting the
app is enough. The Alembic config (`alembic.ini`, `migrations/env.py`) reads
the database URL from `app.core.config` settings (`DATABASE_URL` / `.env`),
so it always targets the same database the app itself would use.

### Creating a new migration

1. Change the models in `app/db/models.py`.
2. Generate a revision from the model diff:

   ```bash
   uv run alembic revision --autogenerate -m "add some_column to some_table"
   ```

3. Open the generated file under `migrations/versions/` and review it —
   autogenerate is a good first draft, not a guarantee. Pay particular
   attention to SQLite limitations (renames, type changes, dropping
   columns): migrations are generated with `render_as_batch=True`, so
   Alembic rewrites the table via a batch (copy-and-swap) instead of a plain
   `ALTER TABLE`, but data-preserving intent (defaults, backfills) is still
   your responsibility to check.
4. Apply it locally and confirm the app still starts cleanly:

   ```bash
   uv run alembic upgrade head
   uv run pytest
   ```

5. Commit the migration file alongside the model change.

### Other useful commands

```bash
uv run alembic current        # show the revision the DB is stamped at
uv run alembic history        # list all revisions
uv run alembic downgrade -1    # roll back one revision (local dev only)
```

## The `orinth` CLI

Ships with the backend — `[project.scripts]` in `pyproject.toml` — so `uv run orinth` works with no
extra install. It is an HTTP client of a running server, not a second implementation of the
platform, so start one first (`make dev`, or `orinth serve`).

```bash
uv run orinth --help
uv run orinth doctor                  # where am I pointed, and is anything there
uv run orinth dataset ls
uv run orinth dataset ingest ./data --prep
uv run orinth dataset readiness <id>  # exits 3 when the dataset is not trainable
```

Code lives in `app/cli/`. Adding a command group is three edits: a row in `main.GROUPS` (a module
*path string*, never an import), a module with `build_parser()` and `run()`, and a row in
`commands/completion.VERBS`. `tests/test_cli_dispatch.py` asserts that `orinth --help` imports no
command module and nothing heavy — that test is what keeps startup fast as commands are added.

## The `orinth` package (notebooks)

`orinth/` is the SDK a notebook kernel imports. It may import `app`; **`app` must never import
it** — an arrow the other way would pull the notebook stack into the API process.
`tests/test_orinth_sdk.py` asserts it.

Requires the optional extra:

```bash
uv sync --extra notebooks
```

Reads (`datasets.load`, `models.predictor`) go straight to the filesystem through `Settings`.
Writes (`datasets.register`) go over HTTP to `/api/datasets/ingest` and `/prep/apply`, so the
project gate, the manifest write, and readiness all run exactly once, in the code that already
owns them.
