"""Phase 15 llama.cpp serving: port allocation, single-session semantics,
state-file orphan reaping, idle-timeout activity accounting, and SSE chat
proxy framing with the upstream mocked."""

import asyncio
import json
import socket
import subprocess
import sys
import time
import types

import pytest

from app.api.routers.serving import UpstreamError, chat_event_stream, upstream_chat_body
from app.ml.model_registry import ModelRegistry
from app.schemas import ChatMessage, ChatRequest, ServingStartRequest
from app.services.serving import ServingError, ServingService


class DummyProcess:
    def __init__(self, pid=4242):
        self.pid = pid
        self.terminated = False
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.terminated = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def registry(settings, storage) -> ModelRegistry:
    return ModelRegistry(settings, storage)


def register_gguf(registry, storage, model_id="gguf1"):
    model_dir = storage.trained_models / f"served-{model_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = model_dir / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    registry.register_model(
        model_id=model_id,
        name=f"GGUF {model_id}",
        family="llm_gguf",
        task_type="llm_finetune",
        paths={"model": gguf_path},
        labels=[],
        format="gguf",
        artifacts={"model_dir": str(model_dir)},
    )


def make_service(settings, storage, registry, monkeypatch, spawned=None):
    """A ServingService whose subprocess spawn and readiness probe are faked."""
    service = ServingService(settings, storage, registry)

    def fake_spawn(self, *, model_path, port, context_length, n_gpu_layers, log_tail):
        process = DummyProcess(pid=5000 + len(spawned or []))
        if spawned is not None:
            spawned.append({"model_path": model_path, "port": port, "process": process})
        return process

    monkeypatch.setattr(ServingService, "_spawn", fake_spawn)
    monkeypatch.setattr(ServingService, "_probe_ready", lambda self, session: None)
    return service


# -- validation ---------------------------------------------------------------


def test_only_gguf_models_are_servable(settings, storage, registry, monkeypatch):
    adapter_dir = storage.trained_models / "adp" / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    registry.register_model(
        model_id="adp",
        name="Adapter",
        family="llm_adapter",
        task_type="llm_finetune",
        paths={"model": adapter_dir},
        labels=[],
    )
    service = make_service(settings, storage, registry, monkeypatch)
    with pytest.raises(ServingError, match="Export this model to GGUF"):
        service.start(ServingStartRequest(model_id="adp"))


# -- port allocation ----------------------------------------------------------


def test_port_allocation_skips_taken_ports(make_settings, storage, registry):
    settings = make_settings(SERVING_PORT_RANGE="8650-8652")
    service = ServingService(settings, storage, registry)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 8650))
        blocker.listen(1)
        assert service._allocate_port() == 8651


def test_port_range_exhaustion_names_setting(make_settings, storage, registry):
    settings = make_settings(SERVING_PORT_RANGE="8653-8654")
    service = ServingService(settings, storage, registry)
    sockets = []
    try:
        for port in (8653, 8654):
            blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            blocker.bind(("127.0.0.1", port))
            blocker.listen(1)
            sockets.append(blocker)
        with pytest.raises(ServingError, match="SERVING_PORT_RANGE"):
            service._allocate_port()
    finally:
        for blocker in sockets:
            blocker.close()


# -- single-session semantics -------------------------------------------------


def test_start_stops_previous_session(settings, storage, registry, monkeypatch):
    register_gguf(registry, storage, "gguf1")
    register_gguf(registry, storage, "gguf2")
    spawned = []
    service = make_service(settings, storage, registry, monkeypatch, spawned)

    first = service.start(ServingStartRequest(model_id="gguf1"))
    assert first.state == "running"
    assert first.model_id == "gguf1"

    second = service.start(ServingStartRequest(model_id="gguf2"))
    assert second.state == "running"
    assert second.model_id == "gguf2"
    assert spawned[0]["process"].terminated, "previous session must be stopped first"
    service.stop()


def test_start_same_model_reuses_session(settings, storage, registry, monkeypatch):
    register_gguf(registry, storage, "gguf1")
    spawned = []
    service = make_service(settings, storage, registry, monkeypatch, spawned)
    service.start(ServingStartRequest(model_id="gguf1"))
    service.start(ServingStartRequest(model_id="gguf1"))
    assert len(spawned) == 1, "concurrent/duplicate start must not respawn"
    service.stop()


def test_stop_clears_state_and_file(settings, storage, registry, monkeypatch):
    register_gguf(registry, storage, "gguf1")
    spawned = []
    service = make_service(settings, storage, registry, monkeypatch, spawned)
    service.start(ServingStartRequest(model_id="gguf1"))
    assert storage.serving_state_file.exists()
    state = json.loads(storage.serving_state_file.read_text(encoding="utf-8"))
    assert state["model_id"] == "gguf1"
    assert state["pid"] == spawned[0]["process"].pid

    status = service.stop()
    assert status.state == "stopped"
    assert spawned[0]["process"].terminated
    assert not storage.serving_state_file.exists()


def test_failed_start_surfaces_stderr_tail(settings, storage, registry, monkeypatch):
    register_gguf(registry, storage, "gguf1")
    service = ServingService(settings, storage, registry)

    def fake_spawn(self, *, model_path, port, context_length, n_gpu_layers, log_tail):
        log_tail.append("llama_model_load: error loading model")
        process = DummyProcess()
        process.returncode = 1
        return process

    monkeypatch.setattr(ServingService, "_spawn", fake_spawn)
    with pytest.raises(ServingError, match="error loading model"):
        service.start(ServingStartRequest(model_id="gguf1"))
    status = service.status()
    assert status.state == "stopped"
    assert any("error loading model" in line for line in status.stderr_tail)
    assert not storage.serving_state_file.exists()


# -- orphan reaping -----------------------------------------------------------


def test_reap_orphans_terminates_recorded_pid(settings, storage, registry):
    process = subprocess.Popen(["sleep", "30"])
    storage.serving.mkdir(parents=True, exist_ok=True)
    storage.serving_state_file.write_text(
        json.dumps({"pid": process.pid, "port": 8600, "model_id": "gguf1"}), encoding="utf-8"
    )
    service = ServingService(settings, storage, registry)
    service.reap_orphans()
    deadline = time.monotonic() + 5
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert process.poll() is not None, "orphaned server must be terminated"
    assert not storage.serving_state_file.exists()


def test_reap_orphans_tolerates_missing_or_bad_state(settings, storage, registry):
    service = ServingService(settings, storage, registry)
    service.reap_orphans()  # no file
    storage.serving.mkdir(parents=True, exist_ok=True)
    storage.serving_state_file.write_text("not json", encoding="utf-8")
    service.reap_orphans()
    assert not storage.serving_state_file.exists()


# -- idle timeout -------------------------------------------------------------


def test_idle_timeout_accounting(make_settings, storage, registry, monkeypatch):
    settings = make_settings(SERVING_IDLE_TIMEOUT_SECONDS="60")
    register_gguf(registry, storage, "gguf1")
    service = make_service(settings, storage, registry, monkeypatch)
    service.start(ServingStartRequest(model_id="gguf1"))
    session = service._session
    assert not service._idle_expired(session, now=session.last_activity + 59)
    assert service._idle_expired(session, now=session.last_activity + 61)

    # Per-token activity updates keep a long generation from counting as idle:
    # a stale timestamp reads as expired until a token touch refreshes it.
    session.last_activity -= 100
    assert service._idle_expired(session)
    service.touch_activity()
    assert not service._idle_expired(session)
    service.stop()


def test_monitor_tick_stops_idle_session(make_settings, storage, registry, monkeypatch):
    settings = make_settings(SERVING_IDLE_TIMEOUT_SECONDS="1")
    register_gguf(registry, storage, "gguf1")
    spawned = []
    service = make_service(settings, storage, registry, monkeypatch, spawned)
    service.start(ServingStartRequest(model_id="gguf1"))
    service._session.last_activity -= 5
    assert service._monitor_tick() == "exit"
    assert service.status().state == "stopped"
    assert spawned[0]["process"].terminated


def test_monitor_tick_reaps_crashed_server(settings, storage, registry, monkeypatch):
    register_gguf(registry, storage, "gguf1")
    spawned = []
    service = make_service(settings, storage, registry, monkeypatch, spawned)
    service.start(ServingStartRequest(model_id="gguf1"))
    spawned[0]["process"].returncode = 137
    assert service._monitor_tick() == "exit"
    status = service.status()
    assert status.state == "stopped"
    assert "exited unexpectedly" in (status.error or "")
    assert not storage.serving_state_file.exists()


# -- SSE chat proxy framing ---------------------------------------------------


def collect_events(upstream, service):
    async def _collect():
        return [event async for event in chat_event_stream(upstream, service)]

    return asyncio.run(_collect())


def make_idle_service(settings, storage, registry):
    return ServingService(settings, storage, registry)


def upstream_from_lines(lines):
    async def _gen():
        for line in lines:
            yield line

    return _gen()


def test_chat_stream_reframes_deltas_and_usage(settings, storage, registry):
    service = make_idle_service(settings, storage, registry)
    chunk1 = json.dumps({"choices": [{"delta": {"content": "Hel"}}]})
    chunk2 = json.dumps({"choices": [{"delta": {"content": "lo"}}]})
    final = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    events = collect_events(
        upstream_from_lines([f"data: {chunk1}", "", f"data: {chunk2}", f"data: {final}", "data: [DONE]"]),
        service,
    )
    payloads = [json.loads(event[len("data: ") :]) for event in events[:-1]]
    assert payloads[0] == {"type": "delta", "content": "Hel"}
    assert payloads[1] == {"type": "delta", "content": "lo"}
    usage = payloads[2]
    assert usage["type"] == "usage"
    assert usage["completion_tokens"] == 2
    assert usage["tokens_per_second"] >= 0
    assert events[-1] == "data: [DONE]\n\n"


def test_chat_stream_upstream_death_emits_error_event(settings, storage, registry):
    service = make_idle_service(settings, storage, registry)

    async def dying_upstream():
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "He"}}]})
        raise UpstreamError("connection reset by peer")

    events = collect_events(dying_upstream(), service)
    payloads = [json.loads(event[len("data: ") :]) for event in events[:-1]]
    assert payloads[0]["type"] == "delta"
    assert payloads[1]["type"] == "error"
    assert "connection reset" in payloads[1]["message"]
    assert events[-1] == "data: [DONE]\n\n"


def test_chat_stream_touches_activity_per_token(settings, storage, registry, monkeypatch):
    register_gguf(registry, storage, "gguf1")
    spawned = []
    service = make_service(settings, storage, registry, monkeypatch, spawned)
    service.start(ServingStartRequest(model_id="gguf1"))
    session = service._session
    session.last_activity -= 100
    stale = session.last_activity
    chunk = json.dumps({"choices": [{"delta": {"content": "x"}}]})
    collect_events(upstream_from_lines([f"data: {chunk}", "data: [DONE]"]), service)
    assert session.last_activity > stale
    service.stop()


def test_upstream_chat_body_prepends_system_and_sampler_params():
    payload = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")],
        system="be brief",
        temperature=0.5,
        top_p=0.9,
        max_tokens=128,
    )
    body = upstream_chat_body(payload)
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert body["stream"] is True
    assert (body["temperature"], body["top_p"], body["max_tokens"]) == (0.5, 0.9, 128)


def test_chat_requires_running_session(settings, storage, registry):
    service = ServingService(settings, storage, registry)
    with pytest.raises(ServingError, match="Start serving first"):
        service.running_session_port()


# -- custom GGUF path serving + scan ------------------------------------------


def test_start_requires_exactly_one_target(settings, storage, registry, monkeypatch):
    service = make_service(settings, storage, registry, monkeypatch)
    with pytest.raises(ServingError, match="exactly one"):
        service.start(ServingStartRequest())
    with pytest.raises(ServingError, match="exactly one"):
        service.start(ServingStartRequest(model_id="a", model_path="/tmp/x.gguf"))


def test_start_by_custom_path(settings, storage, registry, monkeypatch, tmp_path):
    gguf = tmp_path / "my-model.gguf"
    gguf.write_bytes(b"GGUF")
    spawned = []
    service = make_service(settings, storage, registry, monkeypatch, spawned)
    status = service.start(ServingStartRequest(model_path=str(gguf)))
    assert status.state == "running"
    assert status.model_name == "my-model"
    assert spawned[0]["model_path"] == gguf.resolve()
    service.stop()


def test_start_by_path_rejects_non_gguf(settings, storage, registry, monkeypatch, tmp_path):
    other = tmp_path / "weights.safetensors"
    other.write_bytes(b"x")
    service = make_service(settings, storage, registry, monkeypatch)
    with pytest.raises(ServingError, match=".gguf"):
        service.start(ServingStartRequest(model_path=str(other)))


def test_scan_directory_finds_gguf_and_tags_registered(settings, storage, registry, tmp_path):
    # A registered GGUF (under storage) plus a loose one in a scanned dir.
    register_gguf(registry, storage, "reg1")
    registered_path = registry.get_spec("reg1").paths["model"]
    loose_dir = tmp_path / "downloads"
    loose_dir.mkdir()
    (loose_dir / "loose.gguf").write_bytes(b"GGUF")
    # A non-GGUF HF dir to surface as an export candidate.
    hf_dir = loose_dir / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_bytes(b"w")

    service = ServingService(settings, storage, registry)
    result = service.scan_models(str(loose_dir))
    names = {entry.name for entry in result.entries}
    assert "loose.gguf" in names
    assert any(str(hf_dir.resolve()) == d for d in result.exportable_dirs)

    # Scanning the registered file directly tags it with its catalog id.
    single = service.scan_models(str(registered_path))
    assert len(single.entries) == 1
    assert single.entries[0].kind == "registered"
    assert single.entries[0].model_id == "reg1"


def test_scan_missing_path_errors(settings, storage, registry):
    service = ServingService(settings, storage, registry)
    with pytest.raises(ServingError, match="does not exist"):
        service.scan_models("/no/such/dir/xyz")


def test_browse_lists_dirs_and_gguf(settings, storage, registry, tmp_path):
    root = tmp_path / "browse_root"
    root.mkdir()
    (root / "sub").mkdir()
    (root / ".hidden").mkdir()
    (root / "a.gguf").write_bytes(b"GGUF")
    (root / "notes.txt").write_text("x", encoding="utf-8")
    service = ServingService(settings, storage, registry)
    result = service.browse(str(root))
    assert {d.name for d in result.dirs} == {"sub"}  # hidden dir skipped, txt ignored
    assert [f.name for f in result.gguf_files] == ["a.gguf"]
    assert result.parent == str(root.resolve().parent)


def test_browse_defaults_to_home_and_rejects_files(settings, storage, registry, tmp_path):
    service = ServingService(settings, storage, registry)
    # No path → home directory (exists, is a dir).
    assert service.browse(None).path
    file_path = tmp_path / "x.gguf"
    file_path.write_bytes(b"GGUF")
    with pytest.raises(ServingError, match="Not a directory"):
        service.browse(str(file_path))


# -- config persistence, native picker, HF catalog (phase 15 additions) -------


def test_config_persists_and_drops_dead_dir(settings, storage, registry, tmp_path):
    from app.schemas import ServingConfigUpdate

    service = ServingService(settings, storage, registry)
    assert service.load_config().models_dir is None

    real_dir = tmp_path / "models"
    real_dir.mkdir()
    saved = service.save_config(ServingConfigUpdate(models_dir=str(real_dir)))
    assert saved.models_dir == str(real_dir.resolve())
    assert service.load_config().models_dir == str(real_dir.resolve())

    # A saved dir that no longer exists is not restored.
    real_dir.rmdir()
    assert service.load_config().models_dir is None


def test_save_config_rejects_non_directory(settings, storage, registry, tmp_path):
    from app.schemas import ServingConfigUpdate

    service = ServingService(settings, storage, registry)
    with pytest.raises(ServingError):
        service.save_config(ServingConfigUpdate(models_dir=str(tmp_path / "nope")))


def test_pick_path_unavailable_when_dialog_tool_missing(settings, storage, registry, monkeypatch):
    import app.services.serving as serving_module

    service = ServingService(settings, storage, registry)
    monkeypatch.setattr(serving_module.platform, "system", lambda: "Darwin")

    def boom(*args, **kwargs):
        raise FileNotFoundError("osascript")

    monkeypatch.setattr(serving_module.subprocess, "run", boom)
    result = service.pick_path("folder")
    assert result.unavailable is True and result.path is None


def test_pick_path_returns_chosen_folder(settings, storage, registry, monkeypatch):
    import app.services.serving as serving_module

    service = ServingService(settings, storage, registry)
    monkeypatch.setattr(serving_module.platform, "system", lambda: "Darwin")

    class _Proc:
        returncode = 0
        stdout = "/Volumes/SSD/models\n"
        stderr = ""

    monkeypatch.setattr(serving_module.subprocess, "run", lambda *a, **k: _Proc())
    result = service.pick_path("folder")
    assert result.path == "/Volumes/SSD/models" and not result.canceled


def test_pick_path_canceled(settings, storage, registry, monkeypatch):
    import app.services.serving as serving_module

    service = ServingService(settings, storage, registry)
    monkeypatch.setattr(serving_module.platform, "system", lambda: "Darwin")

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "execution error: User canceled. (-128)"

    monkeypatch.setattr(serving_module.subprocess, "run", lambda *a, **k: _Proc())
    assert service.pick_path("folder").canceled is True


def test_quant_from_filename():
    from app.services.serving import _quant_from_filename

    assert _quant_from_filename("gemma-4-12b-Q4_K_M.gguf") == "Q4_K_M"
    assert _quant_from_filename("model.Q8_0.gguf") == "Q8_0"
    assert _quant_from_filename("plain.gguf") is None


def test_search_hub_surfaces_error_not_500(settings, storage, registry, monkeypatch):
    service = ServingService(settings, storage, registry)

    class _Api:
        def __init__(self, **kwargs):
            pass

        def list_models(self, **kwargs):
            raise RuntimeError("hub offline")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    response = service.search_hub("gemma", "gguf")
    assert response.error and response.results == []


def test_download_hub_gated_message(settings, storage, registry, monkeypatch):
    service = ServingService(settings, storage, registry)
    import huggingface_hub

    def gated(*args, **kwargs):
        raise RuntimeError("403 Client Error: gated repo")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", gated)
    with pytest.raises(ServingError, match="gated"):
        service.download_hub("google/gemma-3-4b-it-gguf", "model.gguf")


def test_web_search_maps_hits_and_tolerates_failure(settings, storage, registry, monkeypatch):
    service = ServingService(settings, storage, registry)

    class _DDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=5):
            return [
                {"title": "Claude", "href": "https://claude.com", "body": "AI assistant"},
                {"title": "No URL", "body": "skipped"},
            ]

    # `ddgs` is an optional dep absent from the fast CI job; inject a stand-in
    # module so the service's lazy `from ddgs import DDGS` resolves to the fake.
    fake_ddgs = types.ModuleType("ddgs")
    fake_ddgs.DDGS = _DDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_ddgs)
    response = service.web_search("claude", max_results=5)
    assert response.error is None
    assert [r.url for r in response.results] == ["https://claude.com"]
    assert response.results[0].snippet == "AI assistant"

    def boom(*a, **k):
        raise RuntimeError("ratelimited")

    fake_ddgs.DDGS = boom
    failed = service.web_search("claude")
    assert failed.error and failed.results == []


def test_web_search_empty_query_returns_nothing(settings, storage, registry):
    service = ServingService(settings, storage, registry)
    assert service.web_search("  ").results == []


def test_recommended_models_rank_by_fit(settings, storage, registry, monkeypatch):
    service = ServingService(settings, storage, registry)
    monkeypatch.setattr(ServingService, "_detect_memory", lambda self: ("mps", 16.0))
    response = service.recommended_hub_models()
    assert response.total_memory_gb == 16.0
    assert response.results, "curated recommendations must be non-empty"
    # Fitting models come first; the 32B never fits on 16 GB.
    assert response.results[0].fits is True
    big = next(r for r in response.results if r.params == "32B")
    assert big.fits is False
    # All repos are downloadable GGUF repos.
    assert all(r.repo_id.endswith("-GGUF") for r in response.results)


def test_recommended_models_tiny_machine_flags_most_as_tight(settings, storage, registry, monkeypatch):
    service = ServingService(settings, storage, registry)
    monkeypatch.setattr(ServingService, "_detect_memory", lambda self: ("cpu", 4.0))
    response = service.recommended_hub_models()
    # On 4 GB, at least the smallest fits and the large ones do not.
    assert any(r.fits for r in response.results)
    assert any(not r.fits for r in response.results)


def test_search_hub_does_not_pass_direction(settings, storage, registry, monkeypatch):
    service = ServingService(settings, storage, registry)

    class _Api:
        def __init__(self, **kwargs):
            pass

        def list_models(self, **kwargs):
            assert "direction" not in kwargs, "huggingface_hub 1.x rejects `direction`"
            return []

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    assert service.search_hub("gemma", "gguf").error is None
