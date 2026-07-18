# AI Rules

## Source Control

- Track app source, docs, specs, and lockfiles.
- Ignore large or generated assets through `.gitignore`.
- Do not remove user data or generated research assets unless explicitly asked.

## Backend

- Use `uv` and Python 3.11.
- Keep `backend/app/api/routes.py` as a small aggregator over domain routers in `backend/app/api/routers/`.
- Keep shared runtime service instances in `backend/app/container.py`, not in route modules.
- Keep API schemas in `backend/app/schemas.py`.
- Keep DB models in `backend/app/db/models.py`.
- Preserve service package compatibility imports, for example `from app.services.datasets import DatasetService`.
- Keep settings path-safe and relative to the workspace root.
- Keep predictors behind `backend/app/ml/predictors`.
- Avoid importing TensorFlow, Ultralytics, OpenCV, or PyTorch at module import unless unavoidable.
- Manage schema changes with Alembic migrations under `backend/migrations/`, not `Base.metadata.create_all` or hand-written `ALTER TABLE` patches. When you change `backend/app/db/models.py`, generate a matching migration with `uv run alembic revision --autogenerate -m "..."`, review it, and commit it alongside the model change. See `backend/README.md` for the full workflow.

## Frontend

- Use pnpm.
- Keep route files under `frontend/app` as thin entrypoints.
- Keep feature implementations under `frontend/features` and shared page exports in `frontend/components/platform-pages.tsx`.
- Keep API calls in the domain-based `frontend/lib/api/` package and preserve `import { api } from "@/lib/api"`.
- Keep generated TypeScript build info out of git tracking.
- Keep the first screen as the working app, not a landing page.
- Prefer dense, operational UI over marketing-style presentation.
- Use the design tokens defined in `frontend/app/globals.css` and documented in `frontend/DESIGN.md`; never hardcode hex, rgb, or hsl values.
- Use the shared primitives from `@/features/platform/ui` (Button, Badge, EmptyState, toast, skeletons) instead of raw class names in new code.
- Follow the `/frontend-design` skill for UI work. Use `hallmark` only via `audit` / `redesign` or on component-scope briefs; its greenfield flow composes marketing pages and is out of scope here.
- Structure comes from 1px `--line` borders, not shadow. Never add `box-shadow` to a resting panel, card, or row; the only elevations are `--shadow-subtle`, `--shadow-overlay`, and `--ring-accent`.
- Pair status colors as `-tint` background with `-strong` text. The base status hues fail WCAG AA at badge sizes.
- The `--label-0..7` ramp is for data visualization only (annotation classes, EDA bars, ROC curves) and is the sole exception to the single-accent rule. Consume it through `labelColor` / `labelFill`, never by string-concatenating a hex alpha suffix.
- Avoid hiding errors. Surface backend failures in the relevant panel and the toast layer.

## ML and Data

- Class labels are `granuloma`, `kista`, and image-level `Normal` when no lesion is predicted.
- YOLO default confidence is `0.65`; inference IoU default is `0.7`.
- Object-level testing IoU threshold is `0.2`, following notebook references.
- U-Net input is `256x256`; Inception classifier input is `299x299`.

## Security and Privacy

- Never commit secrets.
- Never commit PHI or patient-identifying data.
- Keep uploads and predictions in ignored local storage.
- Add auth before multi-user or network-exposed deployment.
