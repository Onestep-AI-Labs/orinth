"""Following a job that runs on the server, and saying what it is doing.

The requirement this module exists for is that progress **names the stage**. A
prep run is tens of seconds and a training run is minutes to hours; a spinner
for that long is indistinguishable from a hang, and the user's only recourse is
to kill it and hope.

Phase 21 put the answer on the wire: `metadata.prep` carries `step`, `detail`,
and `progress` alongside `state`, and `GET /prep/status` is a single manifest
read costing milliseconds — built precisely so a client can poll it without
loading the thread pool the run is using. So this prints the server's own
sentence and never invents one.

Two rules shape the output:

- **One line per change, not per poll.** Reprinting an unchanged status every
  1.5s fills a CI log with thousands of identical lines. A keepalive covers the
  case where a stage legitimately takes minutes.
- **Ctrl-C detaches, it does not cancel.** The job belongs to the server, which
  is the whole point of being an HTTP client. Interrupting the follower is the
  same gesture as closing the browser tab, and killing an hour of GPU because
  someone wanted their prompt back would be indefensible. The reattach command
  is printed on the way out.
"""

import signal
import time
from collections.abc import Callable
from typing import Any

from app.cli import output
from app.cli.errors import EXIT_INTERRUPTED, CliError

#: Matches the studio's own cadence. `/prep/status` is one JSON read, so this is
#: cheap; the training and evaluation job routes are DB reads of comparable cost.
POLL_SECONDS = 1.5
#: Re-announce an unchanged stage after this long, so a long stage still looks
#: alive without printing on every poll.
KEEPALIVE_SECONDS = 30.0


class Interrupted(CliError):
    def __init__(self, reattach: str) -> None:
        super().__init__(
            "Interrupted. The job is still running on the server.",
            code=EXIT_INTERRUPTED,
            hint=f"Reattach with: {reattach}",
        )


def follow(
    *,
    poll: Callable[[], dict[str, Any]],
    is_done: Callable[[dict[str, Any]], bool],
    describe: Callable[[dict[str, Any]], str],
    reattach: str,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Poll until done, printing each change to stderr.

    Returns the final status. Raises `Interrupted` on SIGINT — the handler is
    installed only for the duration of the follow, and the previous handler is
    restored, because a library-ish function that permanently rebinds SIGINT
    would surprise anything that called it twice.
    """
    interrupted = {"hit": False}

    def on_sigint(_signum, _frame):
        if interrupted["hit"]:
            # A second Ctrl-C means they want out now, not a tidy message.
            raise KeyboardInterrupt
        interrupted["hit"] = True

    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, on_sigint)
    deadline = time.monotonic() + timeout if timeout else None
    last_line = ""
    last_print = 0.0

    try:
        while True:
            if interrupted["hit"]:
                raise Interrupted(reattach)
            status = poll()
            line = describe(status)
            now = time.monotonic()
            if line and (line != last_line or now - last_print >= KEEPALIVE_SECONDS):
                output.note(line)
                last_line = line
                last_print = now
            if is_done(status):
                return status
            if deadline and now > deadline:
                raise CliError(
                    "Gave up waiting for the job to finish. It is still running on the server.",
                    hint=f"Check on it with: {reattach}",
                )
            time.sleep(POLL_SECONDS)
    finally:
        signal.signal(signal.SIGINT, previous)


def prep_line(status: dict[str, Any]) -> str:
    """One line for a prep run, built from the fields phase 21 added.

    `detail` is the server's sentence and carries the counts ("Read 312 files —
    looks like classification, checking"); `step` is the machine-readable stage.
    Preferring `detail` and falling back to `step` means a server that predates
    those fields still produces something rather than a blank line.
    """
    detail = str(status.get("detail") or "").strip()
    step = str(status.get("step") or status.get("state") or "").strip()
    fraction = status.get("progress")
    prefix = f"[{step}]" if step else ""
    percent = (
        f" {round(float(fraction) * 100)}%"
        if isinstance(fraction, (int, float)) and 0 <= float(fraction) <= 1
        else ""
    )
    body = detail or step
    return f"{prefix}{percent} {body}".strip() if body else ""


#: Hard cap on one progress line. A follow that prints a 4,000-character line
#: per poll is unreadable in a terminal and unusable in a CI log.
MAX_LINE = 160


def job_line(status: dict[str, Any]) -> str:
    """One line for a training / evaluation / inference job row.

    The first version stringified whatever `progress` held. That field is a
    *dict* on training jobs — percent, step, ETA, **and the last log lines** —
    so following a real run printed 195 KB of Keras progress bars, backspace
    characters and all. Found by running one.

    So this reads the fields it knows by name and never falls back to `str()` on
    a container: an unrecognised shape prints nothing, which is the right amount
    of noise for something whose meaning is unknown.
    """
    state = str(status.get("status") or "").strip()
    parts = [f"[{state}]"] if state else []

    progress = status.get("progress")
    if isinstance(progress, (int, float)):
        # A bare number is a 0..1 fraction on some jobs and already a percentage
        # on others; both read correctly under this.
        percent = float(progress) * 100 if float(progress) <= 1 else float(progress)
        parts.append(f"{percent:.0f}%")
    elif isinstance(progress, dict):
        if isinstance(progress.get("percent"), (int, float)):
            parts.append(f"{float(progress['percent']):.0f}%")
        step = progress.get("current_step")
        if isinstance(step, str) and step:
            parts.append(step)
        eta = progress.get("eta_seconds")
        if isinstance(eta, (int, float)) and eta > 0:
            parts.append(f"eta {int(eta)}s")

    for key in ("current_epoch", "message"):
        value = status.get(key)
        # Scalars only. `logs` and every other container stay out of the line.
        if isinstance(value, (str, int, float)) and str(value).strip():
            parts.append(f"epoch {value}" if key == "current_epoch" else str(value))

    line = " ".join(parts).strip()
    return line if len(line) <= MAX_LINE else line[: MAX_LINE - 1] + "…"
