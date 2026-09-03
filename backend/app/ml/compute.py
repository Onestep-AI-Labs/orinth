"""What this machine can actually train on, per framework.

Phase 14's `probe_llm_environment` answers this for `llm_sft` only, and answers
it with one `device` field. That is not enough, and this machine proves why:
**torch reports MPS available while TensorFlow reports CPU only**, because
`tensorflow-metal` is a separate wheel. A user here gets a GPU for LLM
fine-tuning and silently gets CPU for Keras training, and nothing in the app
said so — the first evidence was a run that took twenty times longer than
expected.

So the unit of truth is a *(framework, device)* pair, not a device. Each backend
is probed on its own terms, in a subprocess, and reports what it can see.

**Probing happens out of process.** Importing torch and TensorFlow into the API
to ask them about devices is exactly what `docs/ai/rules.md` forbids, and the
cost is not hypothetical: TF alone is seconds of import and hundreds of
megabytes resident, paid by a FastAPI worker that will never train anything.
The probe runs `sys.executable -c` and parses one JSON line, then caches — a
machine does not grow a GPU between requests.
"""

import json
import os
import platform
import subprocess
import sys
import threading
import time

from app.schemas import ComputeDevice, ComputeEnvironment, ComputeFramework

#: A probe that hangs must not hang the request. TF's first import on a cold
#: filesystem is slow but bounded; past this something is genuinely wrong and
#: reporting "unknown" beats blocking a page load.
PROBE_TIMEOUT_SECONDS = 90
#: A machine does not gain an accelerator while the server runs. Re-probing on
#: every request would pay the import cost forever for an answer that cannot
#: change; the cache is cleared by a restart, which is when it could.
CACHE_SECONDS = 600

_cache: tuple[float, ComputeEnvironment] | None = None
_lock = threading.Lock()


# The probe source runs in a fresh interpreter. It is a string rather than an
# importable module so nothing in `app/` can accidentally import it in-process
# and undo the whole arrangement.
_PROBE = r"""
import json, platform, sys

def torch_probe():
    try:
        import torch
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"[:200]}
    devices = []
    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append({
                    "kind": "cuda",
                    "index": index,
                    "name": props.name,
                    "total_memory_mb": int(props.total_memory / (1024 ** 2)),
                    "capability": f"{props.major}.{props.minor}",
                })
    except Exception:
        pass
    try:
        if not devices and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # Apple silicon shares one pool between CPU and GPU; torch reports
            # the ceiling it will allocate rather than a dedicated VRAM figure.
            memory = None
            try:
                memory = int(torch.mps.recommended_max_memory() / (1024 ** 2))
            except Exception:
                memory = None
            devices.append({
                "kind": "mps",
                "index": 0,
                "name": f"Apple {platform.machine()} GPU",
                "total_memory_mb": memory,
                "capability": None,
            })
    except Exception:
        pass
    return {"available": True, "version": torch.__version__, "devices": devices}


def tensorflow_probe():
    try:
        import tensorflow as tf
    except Exception as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"[:200]}
    devices = []
    try:
        for index, device in enumerate(tf.config.list_physical_devices("GPU")):
            detail = {}
            try:
                detail = tf.config.experimental.get_device_details(device)
            except Exception:
                detail = {}
            devices.append({
                "kind": "cuda" if "metal" not in str(device).lower() else "metal",
                "index": index,
                "name": detail.get("device_name") or device.name,
                "total_memory_mb": None,
                "capability": ".".join(str(part) for part in detail["compute_capability"])
                if detail.get("compute_capability") else None,
            })
    except Exception:
        pass
    return {"available": True, "version": tf.__version__, "devices": devices}


print("@@compute@@" + json.dumps({
    "torch": torch_probe(),
    "tensorflow": tensorflow_probe(),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "machine": platform.machine(),
}))
"""


def _run_probe() -> dict:
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            # TF prints pages of C++ logging to stderr on import; silencing it
            # keeps the failure path readable when there genuinely is one.
            env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Device probe timed out after {PROBE_TIMEOUT_SECONDS}s."}
    except OSError as error:
        return {"error": f"Could not run the device probe: {error}"}

    for line in result.stdout.splitlines():
        if line.startswith("@@compute@@"):
            try:
                return json.loads(line[len("@@compute@@") :])
            except json.JSONDecodeError:
                break
    return {"error": (result.stderr or "The device probe returned nothing.").strip()[:300]}


def _framework(name: str, payload: dict) -> ComputeFramework:
    devices = [
        ComputeDevice(
            kind=entry.get("kind", "cpu"),
            index=int(entry.get("index", 0)),
            name=entry.get("name") or "Unknown device",
            total_memory_mb=entry.get("total_memory_mb"),
            capability=entry.get("capability"),
        )
        for entry in payload.get("devices", [])
    ]
    return ComputeFramework(
        name=name,
        installed=bool(payload.get("available")),
        version=payload.get("version"),
        devices=devices,
        # No GPU means CPU, which is a real answer and not an error.
        accelerated=bool(devices),
        error=payload.get("error"),
    )


def probe(refresh: bool = False) -> ComputeEnvironment:
    global _cache
    with _lock:
        if not refresh and _cache is not None and time.monotonic() - _cache[0] < CACHE_SECONDS:
            return _cache[1]

        raw = _run_probe()
        if "error" in raw and "torch" not in raw:
            environment = ComputeEnvironment(
                python_version=platform.python_version(),
                platform=platform.platform(),
                machine=platform.machine(),
                error=raw["error"],
            )
        else:
            environment = ComputeEnvironment(
                python_version=raw.get("python") or platform.python_version(),
                platform=raw.get("platform") or platform.platform(),
                machine=raw.get("machine") or platform.machine(),
                frameworks=[
                    _framework("torch", raw.get("torch") or {}),
                    _framework("tensorflow", raw.get("tensorflow") or {}),
                ],
            )
            environment.notes = _notes(environment)
        _cache = (time.monotonic(), environment)
        return environment


def _notes(environment: ComputeEnvironment) -> list[str]:
    """Say the things a user would otherwise learn from a slow run.

    The disagreement note is the one that matters. Two frameworks reporting
    different accelerators is invisible until a training job takes twenty times
    longer than the one before it, and the reason (a missing `tensorflow-metal`
    wheel) is not something anyone guesses.
    """
    notes: list[str] = []
    torch_fw = environment.framework("torch")
    tf_fw = environment.framework("tensorflow")

    if torch_fw and not torch_fw.installed:
        notes.append("PyTorch is not importable, so LLM and NLP fine-tuning cannot run.")
    if tf_fw and not tf_fw.installed:
        notes.append("TensorFlow is not importable, so Keras training cannot run.")

    if (
        torch_fw
        and tf_fw
        and torch_fw.installed
        and tf_fw.installed
        and torch_fw.accelerated != tf_fw.accelerated
    ):
        fast = "PyTorch" if torch_fw.accelerated else "TensorFlow"
        slow = "TensorFlow" if torch_fw.accelerated else "PyTorch"
        hint = (
            " Install `tensorflow-metal` to give Keras the same GPU."
            if slow == "TensorFlow" and environment.machine == "arm64"
            else ""
        )
        notes.append(
            f"{fast} sees a GPU here but {slow} does not, so runs on {slow} will use the CPU "
            f"and take far longer.{hint}"
        )

    if environment.accelerators() == []:
        notes.append(
            "No GPU detected. Training will run on the CPU — fine for a smoke test, "
            "impractical for a real run."
        )
    return notes
