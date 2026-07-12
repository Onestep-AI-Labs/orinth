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
