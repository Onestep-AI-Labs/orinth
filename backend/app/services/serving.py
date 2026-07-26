"""Managed llama.cpp serving (phase 15).

One ``ServingService`` singleton owns at most one ``python -m
llama_cpp.server`` subprocess. In-memory state is authoritative during normal
operation; ``storage/serving/state.json`` mirrors just the pid/port so a
crashed API process can never leak a llama.cpp server — startup and shutdown
both reap whatever the file records.

Only ``llm_gguf`` models are servable: llama.cpp is the single serving runtime
(stated memory constraint on a single-user machine), and non-GGUF LLM models
route through export first.
"""

import json
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic, sleep

from app.core.config import Settings
from app.core.storage import Storage
from app.ml.model_registry import ModelRegistry
from app.schemas import (
    HubDownloadResult,
    HubFilesResponse,
    HubModelFile,
    HubModelResult,
    HubModelSearchResponse,
    HubRecommendationsResponse,
    ServingBrowseDir,
    ServingBrowseFile,
    ServingBrowseResult,
    ServingConfig,
    ServingConfigUpdate,
    ServingPickResult,
    ServingScanEntry,
    ServingScanResult,
    ServingStartRequest,
    ServingStatus,
    WebSearchResponse,
)

_STDERR_TAIL_LINES = 40


class ServingError(ValueError):
    """User-facing serving failure (start refused, crash detail, etc.)."""


class _Session:
    """One live llama.cpp server subprocess and its bookkeeping."""

    def __init__(self, model_id: str, model_name: str, port: int, context_length: int) -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.port = port
        self.context_length = context_length
        self.process: subprocess.Popen | None = None
        self.started_at = datetime.now(UTC).replace(tzinfo=None)
        self.started_monotonic = monotonic()
        self.last_activity = monotonic()
        self.log_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)


class ServingService:
    def __init__(self, settings: Settings, storage: Storage, registry: ModelRegistry) -> None:
        self.settings = settings
        self.storage = storage
        self.registry = registry
        self._lock = threading.Lock()
        self._session: _Session | None = None
        self._state = "stopped"
        self._last_error: str | None = None
        self._last_stderr_tail: list[str] = []
        self._monitor: threading.Thread | None = None
        self._monitor_stop = threading.Event()

    # -- public API ---------------------------------------------------------

    def start(self, payload: ServingStartRequest) -> ServingStatus:
        model_id, model_name, model_path = self._resolve_start_target(payload)

        with self._lock:
            # Concurrent starts serialize here; the loser finds the winner's
            # session already serving and simply gets it back.
            current = self._session
            if current is not None and self._state == "running" and current.model_id == model_id:
                return self.status()
            if current is not None:
                self._stop_locked()

            context_length = payload.context_length or 4096
            port = self._allocate_port()
            session = _Session(model_id, model_name, port, context_length)
            self._session = session
            self._state = "starting"
            self._last_error = None
            self._last_stderr_tail = []
            try:
                session.process = self._spawn(
                    model_path=model_path,
                    port=port,
                    context_length=context_length,
                    n_gpu_layers=self._resolve_gpu_layers(payload.n_gpu_layers),
                    log_tail=session.log_tail,
                )
                self._write_state_file(session)
                self._probe_ready(session)
            except BaseException:
                tail = list(session.log_tail)
                self._terminate_session_process(session)
                self._session = None
                self._state = "stopped"
                self._last_stderr_tail = tail
                self._clear_state_file()
                raise
            self._state = "running"
            session.last_activity = monotonic()
            self._start_monitor()
            return self.status()

    def _resolve_start_target(self, payload: ServingStartRequest) -> tuple[str, str, Path]:
        """Return (session id, display name, gguf path) for a start request.

        Two sources: a registered ``llm_gguf`` model (``model_id``) or a loose
        GGUF file anywhere on the local machine (``model_path``). Exactly one
        must be given.
        """
        has_id = bool((payload.model_id or "").strip())
        has_path = bool((payload.model_path or "").strip())
        if has_id == has_path:
            raise ServingError("Provide exactly one of a registered model or a custom GGUF path.")

        if has_path:
            model_path = Path(payload.model_path).expanduser()  # type: ignore[arg-type]
            if not model_path.is_file():
                raise ServingError(f"No file at the given path: {model_path}")
            if model_path.suffix.lower() != ".gguf":
                raise ServingError("Custom serving path must point to a .gguf file.")
            # A stable session id per path so a repeat start of the same file is
            # recognised as the already-running session.
            return f"path:{model_path.resolve()}", model_path.stem, model_path.resolve()

        spec = self.registry.get_spec(payload.model_id)  # KeyError -> 404 upstream
        if spec.family != "llm_gguf":
            raise ServingError(
                "Only GGUF models are servable. Export this model to GGUF first, then "
                "serve the registered GGUF artifact."
            )
        model_path = spec.paths.get("model")
        if model_path is None or not model_path.exists():
            raise ServingError("The GGUF file for this model is missing on disk.")
        return payload.model_id, spec.name, model_path

    def scan_models(self, raw_path: str) -> ServingScanResult:
        """List servable GGUF files at a path (a file or a directory to scan).

        A directory is walked (bounded) for ``*.gguf`` files; Hugging Face model
        directories found along the way are reported separately as export
        candidates, since only GGUF serves directly. Registered GGUF models are
        tagged with their catalog id so the UI can prefer them.
        """
        target = Path((raw_path or "").strip()).expanduser()
        if not target.exists():
            raise ServingError(f"Path does not exist: {target}")

        registered_by_path = self._registered_gguf_paths()
        entries: list[ServingScanEntry] = []
        exportable_dirs: list[str] = []
        truncated = False

        def add_gguf(path: Path) -> None:
            resolved = path.resolve()
            model_id = registered_by_path.get(str(resolved))
            entries.append(
                ServingScanEntry(
                    path=str(resolved),
                    name=path.name,
                    size_bytes=path.stat().st_size if path.exists() else None,
                    kind="registered" if model_id else "file",
                    model_id=model_id,
                )
            )

        if target.is_file():
            if target.suffix.lower() != ".gguf":
                raise ServingError("The selected file is not a .gguf model.")
            add_gguf(target)
        else:
            limit = 200
            for path in sorted(target.rglob("*.gguf")):
                if len(entries) >= limit:
                    truncated = True
                    break
                if path.is_file():
                    add_gguf(path)
            # Flag non-GGUF model dirs (config.json + a safetensors) as export
            # candidates so the UI can route the user to Export.
            for config in sorted(target.rglob("config.json")):
                parent = config.parent
                if any(parent.glob("*.safetensors")):
                    exportable_dirs.append(str(parent.resolve()))
                if len(exportable_dirs) >= 50:
                    break

        return ServingScanResult(
            root=str(target.resolve()),
            entries=entries,
            exportable_dirs=exportable_dirs,
            truncated=truncated,
        )

    def browse(self, raw_path: str | None) -> ServingBrowseResult:
        """List one directory level: subfolders + GGUF files, for a path picker.

        Empty/blank path starts at the user's home directory. Hidden entries are
        skipped. This is a local single-user convenience (localhost serving) and
        lists real server-side paths, since that is what the serving subprocess
        opens.
        """
        target = Path((raw_path or "").strip()).expanduser() if (raw_path or "").strip() else Path.home()
        try:
            target = target.resolve()
        except OSError as exc:
            raise ServingError(f"Cannot resolve path: {exc}") from exc
        if not target.exists() or not target.is_dir():
            raise ServingError(f"Not a directory: {target}")

        registered_by_path = self._registered_gguf_paths()
        dirs: list[ServingBrowseDir] = []
        gguf_files: list[ServingBrowseFile] = []
        try:
            children = sorted(target.iterdir(), key=lambda item: item.name.lower())
        except PermissionError as exc:
            raise ServingError(f"Permission denied reading {target}") from exc
        for child in children:
            if child.name.startswith("."):
                continue
            try:
                if child.is_dir():
                    dirs.append(ServingBrowseDir(name=child.name, path=str(child)))
                elif child.is_file() and child.suffix.lower() == ".gguf":
                    resolved = str(child.resolve())
                    gguf_files.append(
                        ServingBrowseFile(
                            name=child.name,
                            path=resolved,
                            size_bytes=child.stat().st_size,
                            model_id=registered_by_path.get(resolved),
                        )
                    )
            except OSError:
                continue

        parent = str(target.parent) if target.parent != target else None
        return ServingBrowseResult(path=str(target), parent=parent, dirs=dirs, gguf_files=gguf_files)

    # -- persisted config ----------------------------------------------------

    def load_config(self) -> ServingConfig:
        path = self.storage.serving_config_file
        if not path.exists():
            return ServingConfig()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return ServingConfig()
        models_dir = data.get("models_dir") if isinstance(data, dict) else None
        # Drop a saved dir that no longer exists so the UI never restores a dead path.
        if models_dir and not Path(models_dir).expanduser().is_dir():
            models_dir = None
        return ServingConfig(models_dir=models_dir)

    def save_config(self, update: ServingConfigUpdate) -> ServingConfig:
        models_dir = (update.models_dir or "").strip() or None
        if models_dir:
            resolved = Path(models_dir).expanduser()
            if not resolved.is_dir():
                raise ServingError(f"Not a directory: {resolved}")
            models_dir = str(resolved.resolve())
        self.storage.serving.mkdir(parents=True, exist_ok=True)
        self.storage.serving_config_file.write_text(
            json.dumps({"models_dir": models_dir}, indent=2), encoding="utf-8"
        )
        return ServingConfig(models_dir=models_dir)

    # -- native OS path picker ----------------------------------------------

    def pick_path(self, kind: str) -> ServingPickResult:
        """Open the OS's native file/folder dialog and return the chosen path.

        Uses only stdlib ``subprocess`` over the platform's built-in dialog tool
        (macOS ``osascript``, Linux ``zenity``) — no third-party dependency. The
        serving subprocess opens real local paths a browser file input cannot
        supply, so this is the single-user localhost way to get one. When no
        native dialog is available the result is flagged ``unavailable`` and the
        UI falls back to manual path entry.
        """
        system = platform.system()
        try:
            if system == "Darwin":
                if kind == "file":
                    script = 'POSIX path of (choose file of type {"gguf"} with prompt "Select a GGUF model")'
                else:
                    script = 'POSIX path of (choose folder with prompt "Select a models folder")'
                proc = subprocess.run(
                    ["osascript", "-e", script], capture_output=True, text=True, timeout=300
                )
                if proc.returncode != 0:
                    stderr = (proc.stderr or "").strip()
                    if "-128" in stderr or "User canceled" in stderr:
                        return ServingPickResult(canceled=True)
                    return ServingPickResult(unavailable=True, message=stderr or "Dialog failed.")
                chosen = proc.stdout.strip()
                return ServingPickResult(path=chosen) if chosen else ServingPickResult(canceled=True)
            if system == "Linux":
                args = ["zenity", "--file-selection", "--title", "Select a model"]
                args += ["--directory"] if kind != "file" else ["--file-filter=GGUF | *.gguf"]
                proc = subprocess.run(args, capture_output=True, text=True, timeout=300)
                if proc.returncode != 0:
                    return ServingPickResult(canceled=True)
                chosen = proc.stdout.strip()
                return ServingPickResult(path=chosen) if chosen else ServingPickResult(canceled=True)
            return ServingPickResult(
                unavailable=True,
                message="No native file dialog on this platform; paste a path instead.",
            )
        except FileNotFoundError:
            return ServingPickResult(
                unavailable=True,
                message="Native file-dialog tool not found; paste a path instead.",
            )
        except subprocess.TimeoutExpired:
            return ServingPickResult(canceled=True)

    # -- Hugging Face model catalog -----------------------------------------

    def search_hub(self, query: str, fmt: str) -> HubModelSearchResponse:
        """Search the Hub for GGUF or MLX models (like LM Studio's catalog).

        Returns repos only (files are fetched per-repo on expand) so one search
        stays a single fast API call.
        """
        try:
            from huggingface_hub import HfApi  # noqa: PLC0415

            api = HfApi(token=self.settings.huggingface_token)
            tag = "gguf" if fmt == "gguf" else "mlx"
            # `sort="downloads"` is descending by default; `direction` was removed
            # in huggingface_hub 1.x, so passing it raises TypeError.
            models = api.list_models(
                search=(query or "").strip() or None,
                filter=tag,
                sort="downloads",
                limit=30,
            )
            results = [
                HubModelResult(
                    repo_id=model.id,
                    author=getattr(model, "author", None) or model.id.split("/")[0],
                    format="gguf" if fmt == "gguf" else "mlx",
                    downloads=int(getattr(model, "downloads", 0) or 0),
                    likes=int(getattr(model, "likes", 0) or 0),
                    updated_at=str(getattr(model, "last_modified", "") or "") or None,
                )
                for model in models
            ]
            return HubModelSearchResponse(results=results)
        except Exception as exc:  # noqa: BLE001 — surfaced to the picker, never a 500
            return HubModelSearchResponse(error=str(exc))

    def recommended_hub_models(self) -> HubRecommendationsResponse:
        """Curated GGUF chat models ranked by whether they fit this machine.

        Mirrors LM Studio's "recommended for your hardware": a static list of
        widely-available GGUF repos with a Q4_K_M size estimate, marked ``fits``
        against detected memory (unified memory on Apple/CUDA, RAM on CPU) with
        headroom for context. Real, downloadable ``bartowski/*-GGUF`` repos.
        """
        from app.schemas import HubRecommendation  # noqa: PLC0415

        device, total_gb = self._detect_memory()
        # (repo_id, title, params label, params in billions)
        catalog = [
            ("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "Qwen2.5 0.5B Instruct", "0.5B", 0.5),
            ("bartowski/SmolLM2-1.7B-Instruct-GGUF", "SmolLM2 1.7B Instruct", "1.7B", 1.7),
            ("bartowski/Qwen2.5-1.5B-Instruct-GGUF", "Qwen2.5 1.5B Instruct", "1.5B", 1.5),
            ("bartowski/gemma-2-2b-it-GGUF", "Gemma 2 2B Instruct", "2B", 2.0),
            ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama 3.2 3B Instruct", "3B", 3.0),
            ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5 3B Instruct", "3B", 3.0),
            ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5 7B Instruct", "7B", 7.0),
            ("bartowski/Qwen2.5-14B-Instruct-GGUF", "Qwen2.5 14B Instruct", "14B", 14.0),
            ("bartowski/Qwen2.5-32B-Instruct-GGUF", "Qwen2.5 32B Instruct", "32B", 32.0),
        ]
        budget = (total_gb or 8.0) * 0.7  # leave headroom for context + OS
        results = []
        for repo_id, title, params, billions in catalog:
            approx = round(billions * 0.65 + 0.4, 1)  # ~Q4_K_M on disk/in memory
            fits = approx <= budget
            results.append(
                HubRecommendation(
                    repo_id=repo_id,
                    title=title,
                    params=params,
                    approx_size_gb=approx,
                    fits=fits,
                    note="Fits comfortably" if fits else "May exceed available memory",
                )
            )
        # Best fits first, then by size.
        results.sort(key=lambda r: (not r.fits, r.approx_size_gb))
        return HubRecommendationsResponse(device=device, total_memory_gb=total_gb, results=results)

    def _detect_memory(self) -> tuple[str, float | None]:
        """(device, total memory GB). Unified memory on Apple/CUDA stands in for VRAM."""
        device = "cpu"
        total_gb: float | None = None
        try:
            if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
                total_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
        except (ValueError, OSError):
            total_gb = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                device = "cuda"
                props = torch.cuda.get_device_properties(0)
                total_gb = props.total_memory / 1024**3
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = "mps"
        except Exception:  # noqa: BLE001 — torch optional; memory stays the RAM estimate
            pass
        return device, round(total_gb, 1) if total_gb else None

    def hub_files(self, repo_id: str, fmt: str) -> HubFilesResponse:
        """The GGUF/MLX files (with sizes) in one Hub repo, for the download list."""
        suffixes = (".gguf",) if fmt == "gguf" else (".safetensors", ".npz")
        try:
            from huggingface_hub import HfApi  # noqa: PLC0415

            api = HfApi(token=self.settings.huggingface_token)
            info = api.model_info(repo_id, files_metadata=True)
            files = []
            for sibling in info.siblings or []:
                name = sibling.rfilename
                if not name.lower().endswith(suffixes):
                    continue
                files.append(
                    HubModelFile(
                        filename=name,
                        size_bytes=getattr(sibling, "size", None),
                        quantization=_quant_from_filename(name),
                    )
                )
            files.sort(key=lambda f: (f.size_bytes or 0))
            return HubFilesResponse(repo_id=repo_id, files=files)
        except Exception as exc:  # noqa: BLE001
            return HubFilesResponse(repo_id=repo_id, error=str(exc))

    def download_hub(self, repo_id: str, filename: str) -> HubDownloadResult:
        """Download one Hub file into the serving models dir, then it is scannable.

        GGUF files are directly servable; MLX weights are downloaded but the
        llama.cpp runtime cannot serve them, so the result is flagged.
        """
        servable = filename.lower().endswith(".gguf")
        local_dir = self.storage.serving_models / repo_id.replace("/", "__")
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            os.environ["HF_HUB_DISABLE_XET"] = "1"
            path = Path(
                hf_hub_download(
                    repo_id,
                    filename,
                    local_dir=str(local_dir),
                    token=self.settings.huggingface_token,
                )
            ).resolve()
        except Exception as exc:  # noqa: BLE001 — surfaced as a 400 upstream
            message = str(exc)
            if "gated" in message.lower() or "403" in message or "awaiting" in message.lower():
                raise ServingError(
                    f"Access to {repo_id} is gated — accept its license on "
                    f"https://huggingface.co/{repo_id} with your configured token."
                ) from exc
            raise ServingError(f"Download failed: {message}") from exc
        return HubDownloadResult(
            path=str(path),
            size_bytes=path.stat().st_size if path.exists() else None,
            servable=servable,
            message=None
            if servable
            else "MLX weights aren't servable by the llama.cpp runtime; downloaded for reference only.",
        )

    # -- web search (chat "Search" mode) ------------------------------------

    def web_search(self, query: str, max_results: int = 5) -> WebSearchResponse:
        """DuckDuckGo web search for the chat Search mode (no API key).

        Returns title/url/snippet hits the chat surface shows as citations and
        feeds to the model as grounding context. Failures come back as an error
        string, never a 500 — chat still answers without web grounding.
        """
        from app.schemas import WebSearchResult  # noqa: PLC0415

        clean = (query or "").strip()
        if not clean:
            return WebSearchResponse(query=clean, results=[])
        try:
            from ddgs import DDGS  # noqa: PLC0415

            with DDGS() as ddgs:
                hits = list(ddgs.text(clean, max_results=max_results))
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI, never a 500
            return WebSearchResponse(query=clean, error=str(exc))
        results = []
        for hit in hits:
            url = hit.get("href") or hit.get("url") or hit.get("link") or ""
            if not url:
                continue
            results.append(
                WebSearchResult(
                    title=str(hit.get("title") or url),
                    url=str(url),
                    snippet=str(hit.get("body") or hit.get("snippet") or ""),
                )
            )
        return WebSearchResponse(query=clean, results=results)

    def _registered_gguf_paths(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for spec in self.registry.list_specs():
            if spec.family != "llm_gguf":
                continue
            model_path = spec.paths.get("model")
            if model_path is not None:
                mapping[str(model_path.resolve())] = spec.id
        return mapping

    def stop(self) -> ServingStatus:
        with self._lock:
            self._stop_locked()
            return self.status()

    def status(self) -> ServingStatus:
        session = self._session
        if session is None or self._state == "stopped":
            return ServingStatus(
                state="stopped",
                error=self._last_error,
                stderr_tail=self._last_stderr_tail,
            )
        idle_for = max(0.0, monotonic() - session.last_activity)
        last_activity_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=idle_for)
        return ServingStatus(
            state=self._state,  # type: ignore[arg-type]
            model_id=session.model_id,
            model_name=session.model_name,
            port=session.port,
            context_length=session.context_length,
            started_at=session.started_at,
            uptime_seconds=round(monotonic() - session.started_monotonic, 1),
            last_activity_at=last_activity_at,
            stderr_tail=list(session.log_tail)[-10:],
            error=self._last_error,
        )

    def touch_activity(self) -> None:
        """Called per streamed token so a long generation never counts as idle."""
        session = self._session
        if session is not None:
            session.last_activity = monotonic()

    def running_session_port(self) -> int:
        """Port of the running session, for the chat proxy."""
        with self._lock:
            session = self._session
            if session is None or self._state != "running":
                raise ServingError("No model is being served. Start serving first.")
            return session.port

    def record_crash(self, detail: str) -> None:
        """Mark the session dead after an upstream failure observed mid-stream."""
        with self._lock:
            session = self._session
            if session is None:
                return
            if session.process is not None and session.process.poll() is None:
                # Server still alive — a single failed request is not a crash.
                return
            self._last_error = detail
            self._last_stderr_tail = list(session.log_tail)
            self._session = None
            self._state = "stopped"
            self._clear_state_file()

    # -- lifecycle hooks -----------------------------------------------------

    def reap_orphans(self) -> None:
        """Terminate any recorded subprocess from a previous API process.

        An unreadable state file is still removed — leaving it would re-run
        this dead-end on every startup.
        """
        state = self._read_state_file()
        if state is not None:
            pid = state.get("pid")
            if isinstance(pid, int) and pid > 0:
                _terminate_pid(pid)
        self._clear_state_file()

    def shutdown(self) -> None:
        """Lifespan exit: stop the live session and reap anything recorded."""
        with self._lock:
            self._stop_locked()
        self.reap_orphans()

    # -- internals -----------------------------------------------------------

    def _stop_locked(self) -> None:
        session = self._session
        self._monitor_stop.set()
        if session is None:
            self._state = "stopped"
            self._clear_state_file()
            return
        self._state = "stopping"
        self._terminate_session_process(session)
        self._session = None
        self._state = "stopped"
        self._clear_state_file()

    def _terminate_session_process(self, session: _Session) -> None:
        process = session.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _spawn(
        self,
        *,
        model_path: Path,
        port: int,
        context_length: int,
        n_gpu_layers: int,
        log_tail: deque[str],
    ) -> subprocess.Popen:
        command = [
            sys.executable,
            "-m",
            "llama_cpp.server",
            "--model",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--n_ctx",
            str(context_length),
            "--n_gpu_layers",
            str(n_gpu_layers),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            raise ServingError(f"Could not launch llama.cpp server: {exc}") from exc
        reader = threading.Thread(
            target=_pump_output, args=(process, log_tail), name="serving-log-reader", daemon=True
        )
        reader.start()
        return process

    def _probe_ready(self, session: _Session) -> None:
        """Poll the server's /v1/models until it answers, the process dies, or timeout."""
        deadline = monotonic() + self.settings.serving_ready_timeout_seconds
        url = f"http://127.0.0.1:{session.port}/v1/models"
        while monotonic() < deadline:
            process = session.process
            if process is not None and process.poll() is not None:
                tail = "\n".join(list(session.log_tail)[-12:]) or "(no output)"
                raise ServingError(
                    f"llama.cpp server exited during startup (code {process.returncode}). "
                    "If this is an out-of-memory failure, try a smaller quantization "
                    f"(q4_k_m) or a smaller model.\n{tail}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2):  # noqa: S310
                    return
            except (urllib.error.URLError, OSError):
                sleep(0.25)
        raise ServingError(
            f"llama.cpp server did not become ready within "
            f"{self.settings.serving_ready_timeout_seconds}s."
        )

    def _allocate_port(self) -> int:
        start, end = self.settings.serving_ports
        for port in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        raise ServingError(
            f"No free port in the serving range {start}-{end}. Adjust SERVING_PORT_RANGE "
            "or stop whatever occupies those ports."
        )

    def _resolve_gpu_layers(self, requested: int | None) -> int:
        """Auto: offload everything on Metal/CUDA, nothing on plain CPU."""
        if requested is not None:
            return requested
        if sys.platform == "darwin":
            return -1
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return -1
        except ImportError:
            pass
        return 0

    # -- idle timeout ---------------------------------------------------------

    def _start_monitor(self) -> None:
        self._monitor_stop = threading.Event()
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="serving-idle-monitor", daemon=True
        )
        self._monitor.start()

    def _monitor_loop(self) -> None:
        stop_event = self._monitor_stop
        while not stop_event.wait(5.0):
            action = self._monitor_tick()
            if action == "exit":
                return

    def _monitor_tick(self) -> str:
        """One monitor pass: reap a crashed server, stop an idle one."""
        with self._lock:
            session = self._session
            if session is None or self._state != "running":
                return "exit"
            process = session.process
            if process is not None and process.poll() is not None:
                self._last_error = (
                    f"llama.cpp server exited unexpectedly (code {process.returncode})."
                )
                self._last_stderr_tail = list(session.log_tail)
                self._session = None
                self._state = "stopped"
                self._clear_state_file()
                return "exit"
            if self._idle_expired(session):
                self._stop_locked()
                self._last_error = None
                return "exit"
            return "continue"

    def _idle_expired(self, session: _Session, now: float | None = None) -> bool:
        timeout = self.settings.serving_idle_timeout_seconds
        if timeout <= 0:
            return False
        current = monotonic() if now is None else now
        return (current - session.last_activity) > timeout

    # -- state file -----------------------------------------------------------

    def _write_state_file(self, session: _Session) -> None:
        process = session.process
        self.storage.serving.mkdir(parents=True, exist_ok=True)
        self.storage.serving_state_file.write_text(
            json.dumps(
                {
                    "pid": process.pid if process is not None else None,
                    "port": session.port,
                    "model_id": session.model_id,
                    "started_at": session.started_at.isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _read_state_file(self) -> dict | None:
        path = self.storage.serving_state_file
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return state if isinstance(state, dict) else None

    def _clear_state_file(self) -> None:
        try:
            self.storage.serving_state_file.unlink(missing_ok=True)
        except OSError:
            pass


def _quant_from_filename(name: str) -> str | None:
    """Pull a quant tag (Q4_K_M, Q8_0, …) out of a GGUF filename for display."""
    import re  # noqa: PLC0415

    match = re.search(r"(IQ?\d+(?:_[A-Z0-9]+)*|Q\d+(?:_[A-Z0-9]+)*|F16|BF16|F32)", name, re.IGNORECASE)
    return match.group(0).upper() if match else None


def _pump_output(process: subprocess.Popen, log_tail: deque[str]) -> None:
    if process.stdout is None:
        return
    try:
        for line in process.stdout:
            clean = line.rstrip()
            if clean:
                log_tail.append(clean)
    except ValueError:
        # Stream closed during termination.
        pass


def _terminate_pid(pid: int) -> None:
    """SIGTERM then SIGKILL a recorded pid, tolerating an already-dead process."""
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
