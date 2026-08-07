# Claude Code Instructions

Use the same workflow as Codex.

Use the local project skill when available:

- `$orinth`

Read in order:

1. `AGENTS.md`
2. `docs/ai/workflow.md`
3. `docs/ai/rules.md`
4. The relevant file in `specs/`

Claude slash commands are available in `.claude/commands/`:

- `/plan-spec`
- `/implement-spec`
- `/review-change`

Project skills are available in `.claude/skills/`:

- `/frontend-design` — required for any UI or styling work; enforces `frontend/DESIGN.md`.

Follow the repository rules in `AGENTS.md`. Keep specs current when implementation details change.
