"""Where the CLI is pointed, and how it knows.

Five tiers, highest wins, first hit only — no merging across them. Merging is
tempting and wrong: a `.orinth.toml` in a project directory that supplies only
`project` would otherwise silently inherit a `backend` from the user config, and
the resulting "why is it talking to that server" is exactly the failure this
module exists to prevent. Every resolved value therefore remembers which tier it
came from, and `orinth doctor` prints that alongside the value.

**`--workspace` names a project directory, not a storage root.** Worth being
blunt about, because "workspace" in this repo could plausibly mean `STORAGE_DIR`
and deliberately does not. The storage root, the models and datasets directories,
and the database are the *server's* configuration, read by `Settings` from `.env`
exactly as they are today. The CLI never resolves them, never reads them, and
never writes into them — doing so would make it a second owner of the state the
backend owns, which is the whole reason this is an HTTP client. `--workspace`
only says where to start looking for a `.orinth.toml`.
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from app.cli.errors import EXIT_USAGE, CliError

DEFAULT_BACKEND = "http://127.0.0.1:8000"
#: Mirrors `app.core.defaults.DEFAULT_PROJECT_ID`. Duplicated rather than
#: imported: `app.core.defaults` is harmless today, but importing anything from
#: `app.core` is the first step down the path the dispatch test exists to block.
DEFAULT_PROJECT = "default-research-project"

CONFIG_FILENAME = ".orinth.toml"
USER_CONFIG = Path.home() / ".config" / "orinth" / "config.toml"
#: The desktop build has no `PATH` entry and picks an ephemeral port, so it
#: writes its own address here. Without this, `orinth` cannot find a backend the
#: user can plainly see running.
MACOS_APP_CONFIG = (
    Path.home() / "Library" / "Application Support" / "orinth.ai.studio" / "cli.toml"
)


@dataclass(frozen=True)
class Resolved:
    """A value and the tier that produced it."""

    value: str
    source: str


@dataclass(frozen=True)
class CliConfig:
    backend: Resolved
    project: Resolved
    workspace: Path

    @property
    def backend_url(self) -> str:
        return self.backend.value.rstrip("/")


def _read_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        return {}
    except (tomllib.TOMLDecodeError, OSError) as error:
        # A malformed config is a usage error, not a runtime one: nothing was
        # attempted, and the fix is in the file the user just edited.
        raise CliError(
            f"Could not read {path}: {error}",
            code=EXIT_USAGE,
            hint="Fix the file, or pass --backend / --project explicitly.",
        ) from error


def find_project_config(start: Path) -> Path | None:
    """Nearest `.orinth.toml`, walking up to `$HOME` or the root.

    Stopping at `$HOME` is deliberate. Walking past it would let a stray file in
    `/` or `/Users` configure every project on the machine, and a config nobody
    remembers writing is worse than no config.
    """
    home = Path.home().resolve()
    current = start.resolve()
    for candidate in [current, *current.parents]:
        path = candidate / CONFIG_FILENAME
        if path.is_file():
            return path
        if candidate == home:
            break
    return None


def resolve(
    *,
    backend: str | None = None,
    project: str | None = None,
    workspace: str | None = None,
    environ: dict[str, str] | None = None,
) -> CliConfig:
    environ = os.environ if environ is None else environ
    root = Path(workspace or environ.get("ORINTH_WORKSPACE") or Path.cwd())

    project_config_path = find_project_config(root)
    project_config = _read_toml(project_config_path) if project_config_path else {}

    user_config: dict = {}
    user_config_path: Path | None = None
    for candidate in (USER_CONFIG, MACOS_APP_CONFIG):
        if candidate.is_file():
            user_config = _read_toml(candidate)
            user_config_path = candidate
            break

    def pick(flag: str | None, env_key: str, key: str, default: str) -> Resolved:
        if flag:
            return Resolved(flag, f"--{key}")
        from_env = environ.get(env_key)
        if from_env:
            return Resolved(from_env, env_key)
        if key in project_config and project_config_path is not None:
            return Resolved(str(project_config[key]), str(project_config_path))
        if key in user_config and user_config_path is not None:
            return Resolved(str(user_config[key]), str(user_config_path))
        return Resolved(default, "default")

    return CliConfig(
        backend=pick(backend, "ORINTH_BACKEND", "backend", DEFAULT_BACKEND),
        project=pick(project, "ORINTH_PROJECT", "project", DEFAULT_PROJECT),
        workspace=root,
    )
