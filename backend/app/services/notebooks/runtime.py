"""Supervising one `jupyter-server`, and never importing it.

The backend spawns `python -m jupyter_server` bound to loopback and reaches it
only over HTTP. That is what keeps the Jupyter stack — and, one import deeper,
whatever a kernel loads — out of the API process, which `docs/ai/rules.md`
requires and which the training runners and the llama.cpp server already do.

The presence check uses `importlib.util.find_spec`, which locates a module
without executing it. Importing `jupyter_server` to find out whether it is
installed would defeat the entire arrangement in the act of checking.

`jupyterlab` is deliberately **not** required and its UI is never served. Only
`jupyter-server`'s HTTP + WebSocket API is used, which is the part that has a
stable contract; the editor is ours.
"""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from app.core.config import Settings
from app.core.storage import Storage
from app.schemas import NotebookRuntimeStatus

#: Written so a restarted backend can find and reap the subprocess it orphaned.
#: Same mechanism `serving` uses, for the same reason: a stranded server holding
#: a port is worse than a missing one, because the next start fails confusingly.
STATE_FILE = "runtime.json"

INSTALL_HINT = (
    "Notebooks need the `notebooks` extra. Install it with "
    "`cd backend && uv sync --extra notebooks`, then start the runtime again."
)

#: The kernel name registered for notebooks. Named rather than reusing
#: `python3` so a notebook's `kernelspec` records that it wants *this*
#: interpreter, with the workspace environment, not whatever `python3` resolves
#: to on the next machine.
KERNEL_NAME = "orinth"


class NotebookRuntimeError(RuntimeError):
    """The runtime could not start, or is not installed."""


class NotebookRuntime:
    def __init__(self, settings: Settings, storage: Storage) -> None:
        self.settings = settings
        self.storage = storage
        self.process: subprocess.Popen | None = None
        self.port: int | None = None
        self.token: str | None = None
        self.error: str | None = None
        self._state = "stopped"
        self._lock = threading.Lock()

    # ---- presence ------------------------------------------------------

    @staticmethod
    def available() -> bool:
        """Is `jupyter_server` importable — without importing it."""
        try:
            return importlib.util.find_spec("jupyter_server") is not None
        except (ImportError, ValueError):
            return False

    # ---- status --------------------------------------------------------

    def status(self) -> NotebookRuntimeStatus:
        available = self.available()
        if not available:
            return NotebookRuntimeStatus(
                available=False,
                state="stopped",
                install_hint=INSTALL_HINT,
                python_version=sys.version.split()[0],
            )
        alive = self.process is not None and self.process.poll() is None
        if self._state == "running" and not alive:
            # The subprocess died without anyone asking. Report it rather than
            # continuing to claim `running`, which would make every proxied
            # request fail with a connection error and no explanation.
            self._state = "failed"
            self.error = self.error or "The notebook server exited unexpectedly."
        return NotebookRuntimeStatus(
            available=True,
            state=self._state,  # type: ignore[arg-type]
            port=self.port if alive else None,
            python_version=sys.version.split()[0],
            kernel_count=self._kernel_count() if alive else 0,
            error=self.error,
        )

    def _kernel_count(self) -> int:
        try:
            kernels = self.request("GET", "api/kernels")
        except Exception:  # noqa: BLE001 - a status read must never raise
            return 0
        return len(kernels) if isinstance(kernels, list) else 0

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> NotebookRuntimeStatus:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                return self.status()
            if not self.available():
                raise NotebookRuntimeError(INSTALL_HINT)

            self._reap_orphan()
            self.error = None
            self._state = "starting"
            try:
                self._spawn()
                self._wait_ready()
                self._state = "running"
            except Exception as failure:
                self._state = "failed"
                self.error = str(failure)
                self.stop()
                raise
            return self.status()

    def _spawn(self) -> None:
        self.storage.notebooks.mkdir(parents=True, exist_ok=True)
        self._write_kernelspec()
        self.port = self._allocate_port()
        self.token = os.urandom(24).hex()

        environment = {
            **os.environ,
            # The kernel resolves the workspace from these, so a notebook reads
            # exactly the paths the server is using rather than guessing.
            "STORAGE_DIR": str(self.settings.storage_path),
            "MODELS_DIR": str(self.settings.models_path),
            "DATASETS_DIR": str(self.settings.datasets_path),
            "ORINTH_API_BASE": self.settings.api_base_url,
            # Kernelspecs live in the workspace, not in the user's `~/.jupyter`,
            # so uninstalling is `rm -rf storage/`.
            "JUPYTER_PATH": str(self.storage.notebooks_jupyter),
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(Path(__file__).resolve().parents[3]), os.environ.get("PYTHONPATH")])
            ),
        }
        if self.settings.database_url:
            environment["DATABASE_URL"] = self.settings.database_url

        command = [
            sys.executable,
            "-m",
            "jupyter_server",
            f"--ServerApp.port={self.port}",
            "--ServerApp.ip=127.0.0.1",
            f"--ServerApp.root_dir={self.storage.notebooks}",
            # Every URL the server generates for itself is then already correct
            # on the outside, which is why the proxy rewrites nothing.
            "--ServerApp.base_url=/api/notebooks/proxy/",
            f"--ServerApp.token={self.token}",
            "--ServerApp.password=",
            "--ServerApp.open_browser=False",
            # The token is injected server-side on every proxied request, so the
            # browser holds no credential for XSRF to protect — and the check
            # would only reject our own writes.
            "--ServerApp.disable_check_xsrf=True",
            "--ServerApp.allow_origin=*",
            f"--MappingKernelManager.cull_idle_timeout={self.settings.notebook_kernel_idle_timeout_seconds}",
            "--MappingKernelManager.cull_connected=False",
        ]
        self.process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            command,
            cwd=str(self.storage.notebooks),
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._write_state()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.settings.notebook_ready_timeout_seconds
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise NotebookRuntimeError(
                    "The notebook server exited during startup. "
                    "Check that the `notebooks` extra is installed."
                )
            try:
                self.request("GET", "api/status", timeout=2.0)
                return
            except Exception:  # noqa: BLE001 - not up yet is the expected case
                time.sleep(0.25)
        raise NotebookRuntimeError(
            f"The notebook server did not answer within "
            f"{self.settings.notebook_ready_timeout_seconds}s."
        )

    def stop(self) -> NotebookRuntimeStatus:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        self.port = None
        self.token = None
        if self._state != "failed":
            self._state = "stopped"
        self._clear_state()
        return self.status()

    def shutdown(self) -> None:
        """Called from the app lifespan, beside `serving_service.shutdown()`."""
        try:
            self.stop()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass

    # ---- proxying support ----------------------------------------------

    def base_url(self) -> str:
        if self.port is None:
            raise NotebookRuntimeError("The notebook runtime is not running.")
        return f"http://127.0.0.1:{self.port}"

    def headers(self) -> dict[str, str]:
        """The credential, injected outbound only.

        It never travels back to the browser — the same posture the OpenRouter
        key has had since phase 11.
        """
        return {"Authorization": f"token {self.token}"} if self.token else {}

    def request(self, method: str, path: str, timeout: float = 30.0):
        response = httpx.request(
            method,
            f"{self.base_url()}/api/notebooks/proxy/{path.lstrip('/')}",
            headers=self.headers(),
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json() if response.content else None

    # ---- kernelspec ----------------------------------------------------

    def _write_kernelspec(self) -> None:
        """Register a kernel that is this interpreter, with the repo importable.

        Without an explicit spec, `jupyter-server` offers whatever kernels the
        machine happens to have — which will not have `orinth` on the path and
        will not be the environment the platform runs in.
        """
        directory = self.storage.notebooks_jupyter / "kernels" / KERNEL_NAME
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Orinth",
                    "language": "python",
                    "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[3])},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ---- ports and orphans ---------------------------------------------

    def _allocate_port(self) -> int:
        start, end = self.settings.notebook_ports
        for port in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        raise NotebookRuntimeError(
            f"No free port in the notebook range {start}-{end}. "
            "Adjust NOTEBOOK_PORT_RANGE or stop whatever occupies those ports."
        )

    @property
    def _state_path(self) -> Path:
        return self.storage.notebooks / STATE_FILE

    def _write_state(self) -> None:
        try:
            self._state_path.write_text(
                json.dumps({"pid": self.process.pid if self.process else None, "port": self.port}),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _clear_state(self) -> None:
        self._state_path.unlink(missing_ok=True)

    def _reap_orphan(self) -> None:
        """Kill a server this process started before it was restarted.

        A backend restart abandons the subprocess, which keeps the port and
        keeps answering — so the next start either fails to bind or, worse,
        succeeds on a different port while the old one still holds kernels.
        """
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        pid = state.get("pid")
        if not isinstance(pid, int):
            self._clear_state()
            return
        try:
            os.kill(pid, 15)
        except (ProcessLookupError, PermissionError):
            # Already gone, or belongs to someone else — either way not ours to
            # keep tracking.
            pass
        self._clear_state()
