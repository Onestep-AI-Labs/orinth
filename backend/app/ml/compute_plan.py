"""Resolving `device: auto` into what a run will actually do.

The point is not to pick a device — that part is three lines. It is to pick the
*batch size and precision that fit it*, and to say why, before the run starts.
A user who sets batch size 32 on a machine with 12 GB of shared memory does not
find out it was wrong until the run dies forty minutes in with an allocator
error, and the error names bytes rather than the setting that caused them.

Two things this deliberately does not do:

- **It never lowers a batch size the user set explicitly.** An override is a
  decision, and silently halving it would make the form lie about what ran.
  It warns instead, with the number it would have chosen.
- **It does not guess at memory it cannot see.** TensorFlow reports no VRAM and
  Apple silicon reports a ceiling on a pool it shares with everything else on
  the machine. Where the number is unknown the plan says so and keeps the
  family default, rather than deriving a batch size from a figure it invented.
"""

from app.ml.compute import probe
from app.schemas import ComputeEnvironment, ComputePlan

#: Which framework backs each model family. This is the mapping that makes the
#: per-framework probe useful: a Keras family on a machine where only torch has
#: a GPU is a CPU run, and saying so needs both halves.
FRAMEWORK_FOR_FAMILY = {
    "yolo": "torch",
    "unet_inception": "tensorflow",
    "keras_classification": "tensorflow",
    "architecture": "tensorflow",
    "nlp_text_classification": "torch",
    "nlp_summarization": "torch",
    "nlp_question_answering": "torch",
    "llm_sft": "torch",
    "lm": "torch",
    "text": "torch",
}
DEFAULT_FRAMEWORK = "torch"

#: Rough per-sample GPU footprint in MB, by family. These are estimates and are
#: labelled as such wherever they surface: the real figure depends on image
#: size, sequence length, and the specific checkpoint. They are here to keep a
#: default off the obviously-wrong end of the range, not to be precise.
FOOTPRINT_MB = {
    "yolo": 420,
    "unet_inception": 380,
    "keras_classification": 300,
    "architecture": 300,
    "nlp_text_classification": 260,
    "nlp_summarization": 640,
    "nlp_question_answering": 520,
    "llm_sft": 1800,
    "lm": 640,
    "text": 260,
}
DEFAULT_FOOTPRINT_MB = 400

#: Left free for the framework's own allocator, kernels, and fragmentation.
#: Filling the card to the last megabyte is how a run dies at epoch nine.
HEADROOM_FRACTION = 0.25

DEFAULT_BATCH = 8
MIN_BATCH = 1
#: Past this the gain is bounded by data loading rather than compute, and the
#: memory estimate above is too coarse to trust that far out.
MAX_BATCH = 64
#: Nothing on a CPU benefits from a large batch, and a big one just makes the
#: first epoch look hung.
CPU_BATCH = 4


def _round_down_pow2(value: int) -> int:
    """Batch sizes are conventionally powers of two, and kernels prefer them."""
    size = MIN_BATCH
    while size * 2 <= value:
        size *= 2
    return size


def plan_for(
    *,
    task_type: str,
    model_family: str,
    requested_device: str | None = None,
    requested_batch_size: int | None = None,
    environment: ComputeEnvironment | None = None,
) -> ComputePlan:
    environment = environment or probe()
    framework_name = FRAMEWORK_FOR_FAMILY.get(model_family, DEFAULT_FRAMEWORK)
    framework = environment.framework(framework_name)

    reasons: list[str] = []
    warnings: list[str] = []

    if framework is None or not framework.installed:
        return ComputePlan(
            task_type=task_type,
            model_family=model_family,
            framework=framework_name,
            device="cpu",
            device_name="CPU",
            accelerated=False,
            batch_size=requested_batch_size or CPU_BATCH,
            precision="fp32",
            reasons=[f"{framework_name} is not importable on this machine."],
            warnings=[
                f"{model_family} needs {framework_name}. Install it before starting a run."
            ],
        )

    device = framework.devices[0] if framework.devices else None
    requested = (requested_device or "auto").strip().lower()

    if requested and requested not in {"auto", ""}:
        # An explicit device is honoured even when the probe disagrees: the user
        # may know something the probe cannot see, and overriding their choice
        # silently is worse than letting a clear error happen.
        resolved = requested
        accelerated = requested != "cpu"
        device_name = device.name if device and device.kind == requested else requested
        reasons.append(f"Device set to `{requested}` explicitly.")
        if accelerated and (device is None or device.kind != requested):
            warnings.append(
                f"`{requested}` was requested but {framework_name} does not report it here. "
                "The run will fail if it is genuinely absent."
            )
    elif device is not None:
        resolved = device.kind
        accelerated = True
        device_name = device.name
        reasons.append(f"{framework_name} reports {device.name} ({device.kind}).")
    else:
        resolved = "cpu"
        accelerated = False
        device_name = "CPU"
        reasons.append(f"{framework_name} sees no GPU on this machine.")
        warnings.append(
            f"{model_family} will train on the CPU. Fine for a smoke test, impractical for a "
            "real run."
        )

    footprint = FOOTPRINT_MB.get(model_family, DEFAULT_FOOTPRINT_MB)
    memory = device.total_memory_mb if device is not None else None

    if requested_batch_size:
        batch = int(requested_batch_size)
        reasons.append(f"Batch size {batch} set explicitly.")
        suggested = _auto_batch(memory, footprint) if accelerated else CPU_BATCH
        if suggested and batch > suggested:
            # Warn, never lower. An override is a decision, and quietly halving
            # it would make the form lie about what ran.
            warnings.append(
                f"Batch size {batch} may not fit: about {memory} MB usable and roughly "
                f"{footprint} MB per sample for {model_family}. {suggested} is the estimate."
            )
    elif not accelerated:
        batch = CPU_BATCH
        reasons.append(f"CPU run, so batch size {CPU_BATCH} — a larger one buys nothing here.")
    elif memory is None:
        batch = DEFAULT_BATCH
        reasons.append(
            f"{framework_name} does not report device memory, so the family default of "
            f"{DEFAULT_BATCH} stands rather than a guess."
        )
    else:
        batch = _auto_batch(memory, footprint) or DEFAULT_BATCH
        usable = int(memory * (1 - HEADROOM_FRACTION))
        reasons.append(
            f"Batch size {batch} from about {usable} MB usable "
            f"({int(HEADROOM_FRACTION * 100)}% held back) and roughly {footprint} MB per "
            f"sample. Estimate, not a measurement."
        )

    precision, precision_reason = _precision(resolved, device)
    reasons.append(precision_reason)

    return ComputePlan(
        task_type=task_type,
        model_family=model_family,
        framework=framework_name,
        device=resolved,
        device_name=device_name,
        accelerated=accelerated,
        batch_size=max(MIN_BATCH, batch),
        precision=precision,
        reasons=reasons,
        warnings=warnings,
    )


def _auto_batch(memory_mb: int | None, footprint_mb: int) -> int | None:
    if not memory_mb:
        return None
    usable = memory_mb * (1 - HEADROOM_FRACTION)
    raw = int(usable // max(1, footprint_mb))
    if raw < MIN_BATCH:
        return MIN_BATCH
    return min(MAX_BATCH, _round_down_pow2(raw))


def _precision(device_kind: str, device) -> tuple[str, str]:
    """Pick the numeric format, and say what decided it.

    bf16 needs Ampere or newer (compute capability 8.0+); on older CUDA cards
    fp16 is the one that actually speeds anything up. MPS supports fp16 but its
    bf16 support has been unreliable across torch releases, so fp16 is the safe
    default there. CPU gets fp32 because reduced precision on a CPU is slower,
    not faster.
    """
    if device_kind == "cuda":
        capability = getattr(device, "capability", None)
        try:
            major = int(str(capability).split(".")[0]) if capability else 0
        except ValueError:
            major = 0
        if major >= 8:
            return "bf16", f"bf16 — compute capability {capability} supports it."
        return "fp16", (
            f"fp16 — compute capability {capability or 'unknown'} predates bf16 support."
        )
    if device_kind in {"mps", "metal"}:
        return "fp16", "fp16 — Metal supports it; bf16 has been unreliable across torch releases."
    return "fp32", "fp32 — reduced precision on a CPU is slower, not faster."
