"""Running model-written Python over dataset rows, without letting it out.

The prep agent hits a wall on data whose shape no rule anticipates — a feature
table with sixteen clinical yes/no columns, a gradebook, a survey export. There
is no vocabulary that maps those onto a task, and the honest deterministic
answer is "ask the user", which is exactly the friction phase 21 exists to
remove. What *does* generalize is code: a model reads the column profile and
writes the twenty lines that reshape those rows into a training example.

Running generated code needs a boundary. This is it.

**Threat model.** The code is written by a model this codebase prompted, over
data the user just uploaded. The realistic failure is *careless*, not hostile:
a stray `requests.get`, an unbounded loop, a write into the dataset directory,
a `print` of the whole file. Those are what the guards below stop. A determined
escape from an in-process CPython sandbox is not stoppable this way and this
module does not claim otherwise — that needs a container, and if untrusted
third-party code ever runs here, that is what it must get.

**The boundary is the function signature, not just the guards.** The sandboxed
code is a pure `transform(rows) -> rows`: it receives a JSON list, returns a
JSON list, and never learns where the dataset lives. Handing it a directory to
read would have made every guard below load-bearing; handing it a list means
they are defense in depth over a process that has nothing to reach for.

What the child process gets:

- **An import allowlist** (`ALLOWED_IMPORTS`) enforced by a `sys.meta_path`
  hook, plus a purge of the dangerous modules CPython pre-imports at startup.
  No `os`, no `socket`, no `subprocess`, and therefore no network and no third-
  party dataframe library either — the transform is a list of dicts in and a
  list of dicts out, and stdlib covers that.
- **A filesystem gate** on `builtins.open`, scoped to the work directory.
- **Resource ceilings** — CPU seconds, output file size, descriptors, and
  address space on Linux (`RLIMIT_AS` is not set on macOS, where CPython
  reserves enough virtual address space at startup that any workable ceiling
  either fails to bind or kills the interpreter before it runs a line).
- **A scrubbed environment**: no `OPENROUTER_API_KEY`, no `HF_TOKEN`, `HOME`
  and `TMPDIR` pointed at the work directory.
- **A wall clock**, enforced by the parent over a new session so a child that
  forks anyway dies with its process group.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

#: Top-level modules the transform may import. Everything else raises
#: `ImportError` inside the child. Chosen for what row reshaping actually needs:
#: parsing, arithmetic, and text handling. Deliberately no `os`, no `pathlib`,
#: and no numpy/pandas/polars — a dataframe library pulls in `ctypes` and
#: `socket` transitively, so admitting one would quietly readmit everything.
ALLOWED_IMPORTS = frozenset(
    {
        "abc", "base64", "bisect", "calendar", "collections", "contextlib",
        "copy", "dataclasses", "datetime", "decimal", "enum", "fractions",
        "functools", "hashlib", "heapq", "itertools", "json", "math",
        "numbers", "operator", "random", "re", "statistics", "string",
        "textwrap", "time", "types", "typing", "unicodedata", "uuid",
        "warnings",
    }
)

#: Wall clock for one transform. Generous for 20k rows of stdlib string work;
#: short enough that a runaway loop is noticed inside one HTTP request.
DEFAULT_TIMEOUT_SECONDS = 30.0
#: CPU ceiling, set below the wall clock so a busy loop dies by `SIGXCPU` with a
#: reportable error rather than by the parent's kill.
DEFAULT_CPU_SECONDS = 20
#: Address space, Linux only. 1 GiB is far more than list-of-dicts work needs.
DEFAULT_MEMORY_BYTES = 1024 * 1024 * 1024
#: Captured stream cap. The output is shown to the user, and a transform that
#: prints per row would otherwise put a megabyte of noise in the plan.
MAX_STREAM_CHARS = 4000
#: Rows the transform may return, independent of what it was given.
MAX_OUTPUT_ROWS = 100_000


@dataclass
class SandboxResult:
    """What came back, and whether it can be believed."""

    ok: bool
    rows: list[dict] = field(default_factory=list)
    #: Empty when `ok`. One sentence, safe to show a user verbatim.
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


# The child. Written as a string rather than a module so nothing in `app` is
# importable from inside the sandbox: the child runs with `-I`, an empty
# `PYTHONPATH`, and a work directory that holds only these three files.
_DRIVER = r'''
import builtins
import json
import sys

WORKDIR, CPU_SECONDS, MEMORY_BYTES, MAX_ROWS = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
)
ALLOWED = set(json.loads(sys.argv[5]))

_real_open = builtins.open


def _fail(message):
    with _real_open(WORKDIR + "/error.txt", "w", encoding="utf-8") as handle:
        handle.write(str(message)[:4000])
    raise SystemExit(1)


# --- resource ceilings, before anything else runs -----------------------------
try:
    import resource

    def _limit(which, value):
        try:
            _, hard = resource.getrlimit(which)
            ceiling = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(which, (ceiling, hard))
        except (ValueError, OSError):
            pass

    _limit(resource.RLIMIT_CPU, CPU_SECONDS)
    _limit(resource.RLIMIT_FSIZE, 256 * 1024 * 1024)
    _limit(resource.RLIMIT_NOFILE, 64)
    # macOS reserves enough virtual address space at interpreter start that any
    # ceiling low enough to be useful kills the process before user code runs.
    if sys.platform.startswith("linux"):
        _limit(resource.RLIMIT_AS, MEMORY_BYTES)
except ImportError:
    pass

payload = json.loads(_real_open(WORKDIR + "/input.json", encoding="utf-8").read())
source = _real_open(WORKDIR + "/transform.py", encoding="utf-8").read()

# --- import gate --------------------------------------------------------------
# Three steps, and the order is the whole trick.
#
# 1. Pre-import every allowed module, so their C accelerators (`_datetime`,
#    `_random`, `_decimal`) are resolved while imports still work. A strict
#    allowlist that fired on those would break `import datetime` — the gate
#    would block the very stdlib it is meant to permit.
# 2. *Then* purge. Step 1 drags dangerous names in as a side effect (`random`
#    binds `os.urandom`, `uuid` touches `ctypes` on macOS), so purging first
#    would be undone by it. Modules that already hold a reference keep working;
#    only the importable *name* disappears.
# 3. Install the hook, which now covers everything not already resolved —
#    `import x` consults `sys.modules` before `sys.meta_path`, which is why the
#    hook alone was never enough to stop `import os`.
for name in sorted(ALLOWED):
    try:
        __import__(name)
    except ImportError:
        pass

for name in list(sys.modules):
    root = name.split(".")[0]
    if root in {
        "os", "posix", "nt", "socket", "_socket", "ssl", "_ssl", "subprocess",
        "_posixsubprocess", "shutil", "ctypes", "_ctypes", "mmap", "signal",
        "select", "selectors", "fcntl", "tempfile", "pathlib", "resource",
        "multiprocessing", "threading", "_thread", "webbrowser", "pty", "tty",
    }:
        sys.modules.pop(name, None)


class _ImportGate:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in ALLOWED:
            return None
        raise ImportError(
            "`%s` is not available inside the data-prep sandbox. Allowed modules: %s"
            % (root, ", ".join(sorted(ALLOWED)))
        )


sys.meta_path.insert(0, _ImportGate())


# --- filesystem gate ----------------------------------------------------------
def _guarded_open(file, mode="r", *args, **kwargs):
    try:
        target = str(file)
    except Exception:
        raise PermissionError("the data-prep sandbox cannot open that")
    if not target.startswith(WORKDIR):
        raise PermissionError(
            "the data-prep sandbox has no filesystem access; work from `rows` "
            "and return rows"
        )
    return _real_open(file, mode, *args, **kwargs)


builtins.open = _guarded_open

# --- run ----------------------------------------------------------------------
namespace = {"__name__": "prep_transform", "__builtins__": builtins}
try:
    exec(compile(source, "transform.py", "exec"), namespace)
except BaseException as error:
    _fail("the transform failed to load: %s: %s" % (type(error).__name__, error))

entry = namespace.get("transform")
if not callable(entry):
    _fail("the transform must define a top-level function `transform(rows)`")

try:
    result = entry(payload)
except BaseException as error:
    _fail("the transform raised %s: %s" % (type(error).__name__, error))

if not isinstance(result, list):
    _fail("`transform(rows)` must return a list, got %s" % type(result).__name__)
if len(result) > MAX_ROWS:
    _fail("`transform(rows)` returned %d rows; the cap is %d" % (len(result), MAX_ROWS))

builtins.open = _real_open
try:
    with _real_open(WORKDIR + "/output.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, default=str)
except (TypeError, ValueError) as error:
    _fail("the transform returned values that are not JSON: %s" % error)
'''


def run_transform(
    code: str,
    rows: list[dict],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    memory_bytes: int = DEFAULT_MEMORY_BYTES,
    max_output_rows: int = MAX_OUTPUT_ROWS,
) -> SandboxResult:
    """Run `code`'s `transform(rows)` over `rows` and return what it produced.

    Never raises. Every failure — a syntax error, a blocked import, a timeout, a
    non-JSON return — comes back as `ok=False` with a sentence the Prepare tab
    can show, because the caller's answer to all of them is the same: fall back
    to the deterministic transform and say why.
    """
    workdir = Path(tempfile.mkdtemp(prefix="orinth-prep-"))
    try:
        (workdir / "transform.py").write_text(code, encoding="utf-8")
        (workdir / "input.json").write_text(
            json.dumps(rows, ensure_ascii=False, default=str), encoding="utf-8"
        )
        (workdir / "driver.py").write_text(_DRIVER, encoding="utf-8")

        completed = _spawn(
            [
                sys.executable,
                "-I",
                str(workdir / "driver.py"),
                str(workdir),
                str(cpu_seconds),
                str(memory_bytes),
                str(max_output_rows),
                json.dumps(sorted(ALLOWED_IMPORTS)),
            ],
            cwd=workdir,
            timeout_seconds=timeout_seconds,
        )
        if completed is None:
            return SandboxResult(
                ok=False,
                error=(
                    f"The transform did not finish within {timeout_seconds:.0f} seconds "
                    "and was stopped."
                ),
            )

        stdout, stderr, returncode = completed
        reported = _read_capped(workdir / "error.txt")
        output = workdir / "output.json"

        if returncode != 0 or not output.is_file():
            return SandboxResult(
                ok=False,
                error=reported or _first_line(stderr) or "The transform exited without output.",
                stdout=stdout,
                stderr=stderr,
            )

        try:
            parsed = json.loads(output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            return SandboxResult(
                ok=False, error=f"The transform output could not be read ({error}).",
                stdout=stdout, stderr=stderr,
            )

        kept = [row for row in parsed if isinstance(row, dict)]
        if not kept:
            return SandboxResult(
                ok=False,
                error="The transform returned no usable rows.",
                stdout=stdout,
                stderr=stderr,
            )
        return SandboxResult(ok=True, rows=kept, stdout=stdout, stderr=stderr)
    except OSError as error:
        return SandboxResult(ok=False, error=f"The sandbox could not start ({error}).")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _spawn(
    command: list[str], *, cwd: Path, timeout_seconds: float
) -> tuple[str, str, int] | None:
    """Run to completion, or kill the whole process group and return `None`.

    `start_new_session` plus `killpg` rather than `Popen.kill`: a transform that
    somehow spawns a child would otherwise leave it orphaned and still running
    after the timeout fired.
    """
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=_child_env(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_group(process)
        process.communicate()
        return None
    return _cap(stdout), _cap(stderr), process.returncode


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


def _child_env(workdir: Path) -> dict[str, str]:
    """A minimal environment. Inheriting the server's would hand the child every
    API key the platform holds — the one thing a data transform must never see."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
        "LC_ALL": "C.UTF-8",
    }


def _cap(text: str | None) -> str:
    value = text or ""
    if len(value) <= MAX_STREAM_CHARS:
        return value
    return value[:MAX_STREAM_CHARS] + "\n… truncated."


def _read_capped(path: Path) -> str:
    try:
        return _cap(path.read_text(encoding="utf-8", errors="replace")).strip()
    except OSError:
        return ""


def _first_line(text: str) -> str:
    for line in reversed((text or "").strip().splitlines()):
        if line.strip():
            return line.strip()[:400]
    return ""
