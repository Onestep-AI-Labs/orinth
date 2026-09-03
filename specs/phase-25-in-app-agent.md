# Spec: Phase 25 In-App Agent

## Status

**Draft.** Nothing here is built. Written after reading Roboflow's agent
(`docs.roboflow.com/agents/roboflow-agent`) and deciding which half of it fits a
local, single-user platform.

## Goal

An assistant that reaches the whole platform from anywhere in it — a docked
panel over the working app, not a page you navigate to. You ask for something in
the language you were already thinking in ("split this dataset 80/10/10 and
train a MobileNet on it"), it does the work through the same APIs the UI calls,
and it shows you what it did as links you can open.

The measure of success is narrow and testable: **the agent never does anything
the UI cannot do, and never does anything destructive without being asked
twice.** An agent that can only answer questions is a chatbot bolted onto a
studio; an agent that can quietly delete a dataset is a liability.

## Why this shape

Roboflow's agent is the closest existing thing, and three of its decisions are
worth taking:

- **It lives in the shell, not on a page.** Theirs is a sidebar entry; ours is a
  docked panel that any route can open, because the context that makes a request
  cheap to answer ("this dataset", "this run") is the route you are on.
- **It acts, and its edits are drafts until published.** A plan the user reads
  before it runs is the difference between an assistant and a gamble.
- **Long jobs keep running while you chat**, with a count of what is in flight.
  Training is minutes to hours here too; a chat that blocks on it is unusable.

And two that do not transfer:

- **No workspace/permission model.** Orinth is single-user and local
  (`docs/ai/rules.md`: "Add auth before multi-user or network-exposed
  deployment"). Scoped API keys and folder restrictions solve a problem this
  product does not have yet, and building them now would be inventing a threat
  model.
- **No agent-to-agent MCP surface in v1.** Exposing the agent over MCP so other
  assistants can drive Orinth is a real idea and a separate phase; it needs the
  tool layer to be stable first.

## Scope

In:

- A **docked agent panel** reachable from every `(platform)` route, opened by a
  persistent launcher and `Cmd/Ctrl-K`, holding one conversation per project.
- A **tool layer** over the platform's existing HTTP API — the agent calls the
  same endpoints the frontend does, never a service directly.
- **Route context** passed with every message: the current path, the project,
  and the id of whatever the route is about (dataset, model, run, notebook).
- **A plan-then-act loop.** Anything that writes is proposed as a plan the user
  approves; anything that reads runs immediately.
- **Background work**, with in-flight jobs shown as a count on the launcher and
  a message in the thread when one finishes.
- **Citations.** Every claim about the workspace links to the page that shows
  it, so an answer is checkable in one click.

Out:

- Multi-user, permissions, scoped keys. See above.
- An MCP server exposing the agent to other assistants. Later phase.
- Voice, video, or webcam input.
- Autonomous scheduling ("retrain nightly"). A recurring job that nobody watched
  being started by a chat message is a different feature with a different risk
  profile.
- The agent writing notebook *code* into cells. Notebooks already have kernel
  completion; a second code-writing surface competes with it.

## The decisions

### It is a panel, not a route

A route would mean leaving the dataset you are asking about. The panel docks to
the right of `content-shell`, is resizable, and persists open/closed and width
in `localStorage` beside the project sidebar's state. On a narrow viewport it
becomes an overlay sheet — the same breakpoint the notebook rail already uses.

`AppShell` owns it, which is also what makes "from anywhere" true without every
page opting in.

### Route context is structured, not scraped

Every message carries `{path, project_id, entity: {kind, id} | null}`, derived in
the shell from the pathname. The agent gets *facts*, not a screenshot: "this" in
"train on this" resolves to a dataset id before the model ever sees the word.

An agent that had to infer which dataset you meant from the conversation would
be wrong on the first ambiguous message, and the fix — asking every time — is
worse than the bug.

### Tools are HTTP calls to our own API

The tool layer is a registry of typed tools, each one a call to an existing
endpoint. No tool touches `app.services` directly. Three reasons, in order of
weight:

1. **The API is where the rules live** — the project gate, readiness, the
   atomic manifest write. A tool that bypassed it would be a second, weaker
   definition of what a dataset is, exactly as the CLI's transport decision
   argues (`specs/phase-23-cli.md`).
2. It makes every agent action reproducible by hand: the transcript can show the
   request it made.
3. It bounds the blast radius. The agent can do what a browser can do, and
   nothing else.

### Read runs; write asks

Tools are declared `read` or `write`. Reads execute immediately. Writes are
collected into a **plan** — an ordered list of tool calls with their arguments,
rendered as a checklist — and nothing runs until the user approves it. Approving
runs the whole plan; a failure stops it and reports which step.

Destructive tools (`delete_*`, `apply_prep`, `promote_model`) additionally
require the phase-9 `ConfirmationDialog`, typed confirmation and all. Two gates,
because the cost of a wrong delete here is a dataset somebody spent a week
labelling.

### The model is the one already configured

The OpenRouter key and model in settings (phase 11), reused. No new provider, no
new key, and the same rule as everywhere else: **the key never reaches the
client.** A workspace with no key configured gets a panel that says so and links
to Settings, exactly as the prep agent does.

### One conversation per project, on disk

`storage/agent/<project_id>/<conversation_id>.json`: messages, tool calls,
results, and the plan objects. Filesystem-native like notebooks and recipes,
for the same reason — a conversation is a document, and a document that is also
a database row has two truths that drift.

## Interfaces

### API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/agent/conversations` | List, newest first, per project. |
| `POST` | `/api/agent/conversations` | Start one. |
| `GET` | `/api/agent/conversations/{id}` | Full transcript. |
| `DELETE` | `/api/agent/conversations/{id}` | Remove it. |
| `POST` | `/api/agent/conversations/{id}/messages` | Ask. Returns the assistant turn, which is either an answer or a **plan**. |
| `POST` | `/api/agent/conversations/{id}/plans/{plan_id}/approve` | Execute an approved plan. Returns a job id. |
| `POST` | `/api/agent/conversations/{id}/plans/{plan_id}/reject` | Discard it. |
| `GET` | `/api/agent/jobs/{job_id}` | Plan execution progress, reusing `JobProgress`. |
| `GET` | `/api/agent/tools` | The registry, for the panel's "what can it do" list and for tests. |

Streaming: the message endpoint streams tokens over SSE through a Next route
handler, the way `/api/serving/chat` already does — the config rewrite buffers
streaming responses, which is the defect phase 15 hit and documented.

### Schemas

`AgentTool`, `AgentToolCall`, `AgentPlan`, `AgentMessage`, `AgentConversation`,
`AgentContext`, `AgentJobRead`. `AgentTool.kind: Literal["read", "write",
"destructive"]` is the field the whole safety model hangs off, so it is required
and has no default.

### Frontend surfaces

- `frontend/features/agent/` — `agent-panel.tsx` (the dock), `agent-thread.tsx`,
  `agent-plan.tsx` (the approval checklist), `agent-launcher.tsx`, `hooks.ts`,
  `context.ts` (pathname → `AgentContext`).
- `frontend/components/app-shell.tsx` — mounts the launcher and the panel once,
  beside `PlatformTour`.
- `frontend/lib/api/agent.ts`, re-exported from `lib/api/index.ts`.
- `frontend/app/api/agent/stream/route.ts` — the SSE proxy.
- Selectors under `.agent-*` in `platform.css`.

### Backend modules

- `backend/app/services/agent/tools.py` — the registry. One entry per tool:
  name, kind, JSON schema, and the HTTP call it makes.
- `backend/app/services/agent/runner.py` — the loop: context in, tool calls out,
  plan assembly, execution against the approved plan.
- `backend/app/services/agent/store.py` — conversations on disk.
- `backend/app/api/routers/agent.py`, aggregated in `routes.py`.
- `agent_service` in `container.py`.

### Storage

`storage/agent/` (ignored). No DB table, no migration — see the conversation
decision above.

## The v1 tool set

Deliberately small. Every tool is an endpoint that already exists; the work is
the description and the schema, not the plumbing.

**Read** — `list_datasets`, `get_dataset`, `dataset_readiness`, `list_models`,
`list_training_jobs`, `get_training_job`, `list_evaluations`, `get_evaluation`,
`evaluation_per_item`, `list_notebooks`, `list_projects`, `compute_environment`,
`search_docs`.

**Write** — `create_project`, `create_dataset_from_path`, `plan_dataset_prep`,
`start_training`, `start_evaluation`, `run_inference`, `create_notebook`,
`rename_*`.

**Destructive** — `apply_dataset_prep`, `delete_dataset`, `delete_model`,
`delete_training_job`, `promote_model`.

`search_docs` reads the shipped docs and the specs' Goal/Scope sections so "how
do I…" is answered from this repo rather than from the model's memory of some
other product.

## Data Flow

1. The panel sends `{message, context}` to the conversation.
2. The runner builds the prompt: system rules, the tool registry, the context
   block, and the last N turns.
3. The model answers, or requests tools.
4. **Read** tools execute immediately; their results go back into the loop.
5. **Write/destructive** requests are collected into an `AgentPlan` and returned
   unexecuted, with a rendered summary per step.
6. The user approves → the plan runs as a background job, each step through the
   HTTP API, streaming progress into the thread.
7. Every step records its request and response in the transcript.

## Edge cases

- **No OpenRouter key** → the panel explains and links to Settings. Not an error
  toast; it is a configuration state, like the notebook extra.
- **The model asks for a tool that does not exist** → refused, and the refusal
  is fed back so it can correct itself. Never fuzzy-matched to a real tool.
- **A plan step fails** → execution stops there. Steps already applied are
  listed as applied — a partial result the user can see beats a silent rollback
  they cannot.
- **A long job** (training) → the plan step completes when the job is *queued*,
  and the thread subscribes to it. The agent does not hold a request open for an
  hour.
- **The route changes mid-conversation** → the new context rides the next
  message. Earlier turns keep the context they were asked under, so scrollback
  stays honest.
- **A destructive tool on a shared/reference dataset** → refused by the API's
  own rules; the agent surfaces that refusal rather than working around it.
- **A prompt that tries to redefine the agent's rules** — including text that
  arrives inside dataset names, notebook titles, or file contents the agent
  reads — is data, not instruction. Tool *results* are fenced as untrusted in
  the prompt, and no tool result can add tools or approve a plan.

## Acceptance criteria

- The panel opens from every `(platform)` route, and `Cmd-K` focuses it.
- A read question about the current dataset answers with a link to that dataset
  and does not open a plan.
- "Train a MobileNet on this dataset for 10 epochs" produces a plan of exactly
  one `start_training` call with the arguments visible, and starts nothing until
  approved.
- Rejecting a plan leaves the workspace unchanged (asserted by diffing
  `GET /datasets`, `/models`, `/training/jobs` before and after).
- A destructive step additionally raises `ConfirmationDialog`.
- Killing the backend mid-plan leaves the transcript readable and the plan
  marked `interrupted`, never `completed`.
- Tests: the registry has no tool that imports a service (mirroring
  `test_cli_dispatch.py`'s import guard); every tool declares a `kind`; a plan
  containing a write is never auto-executed; tool results cannot inject tools.
- Driven manually against a live backend on a real dataset, per
  `docs/ai/workflow.md` §5 — including one approved plan that actually trains.

## Deferred

- **MCP server**, so external assistants can use these tools. Wants a stable
  registry first.
- **Vision input** (screenshots of a failing chart).
- **Multi-step autonomy** — plans that re-plan after seeing results. v1 plans
  are flat and finite on purpose: a loop that can extend itself is a loop that
  can surprise you, and the approval gate is what makes this safe.
- **Cost display.** Once the agent can spend tokens on the user's key in the
  background, the running total belongs on the launcher.

## Validation

```bash
cd backend && uv run ruff check .
cd backend && uv run pytest
cd frontend && pnpm typecheck && pnpm lint && pnpm test && pnpm build
```

Plus a live run: start the backend, open a dataset page, ask for a training run,
approve the plan, and confirm the job appears on `/training` with the arguments
the plan showed.
