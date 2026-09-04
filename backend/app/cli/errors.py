"""Failure as a value, with the exit code attached.

Every command raises `CliError` rather than calling `sys.exit`, so `main()` is
the single place that decides what reaches the terminal and with what status.
That matters for the codes below: they are a contract a CI job branches on, and
a code chosen at the point of failure drifts the moment two commands disagree
about what "not ready" means.
"""

#: Success.
EXIT_OK = 0
#: A runtime failure: the server said no, a job finished `failed`, a file could
#: not be read, one input of a batch failed.
EXIT_FAILURE = 1
#: A usage error. Matches argparse's own exit status for a parse failure, which
#: is why that is adopted rather than overridden.
EXIT_USAGE = 2
#: The dataset is not trainable. Distinct from `EXIT_FAILURE` because CI treats
#: "your data is not ready" and "the run crashed" differently: the first is a
#: gate to report, the second is an incident.
EXIT_NOT_READY = 3
#: No backend answered. Distinct for the same reason — "the server is down" is
#: an infrastructure problem, not a problem with what was asked.
EXIT_UNREACHABLE = 4
#: SIGINT during a follow. The server-side job is untouched and still running.
EXIT_INTERRUPTED = 130


class CliError(Exception):
    """A failure with a message for the user and a status for the shell.

    `hint` is a second line offering the next concrete action. It is separate
    from the message so `--json` can carry the two apart, and so a caller can
    add one to an error raised deeper down.
    """

    def __init__(self, message: str, *, code: int = EXIT_FAILURE, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.hint = hint


class BackendUnreachable(CliError):
    def __init__(self, url: str, source: str, detail: str = "") -> None:
        super().__init__(
            f"No Orinth backend at {url}" + (f" ({detail})" if detail else ""),
            code=EXIT_UNREACHABLE,
            # Naming where the URL came from is the difference between a user
            # fixing their config and a user restarting a server that was
            # already running on a different port.
            hint=f"URL came from {source}. Start one with `orinth serve`, or pass --backend URL.",
        )
