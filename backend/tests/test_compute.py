"""Device detection and the plan derived from it.

The probe itself is not exercised here — it spawns a subprocess that imports
torch and TensorFlow, which is seconds and would fail on a machine missing
either. What is worth pinning down is the reasoning built on top of it, and that
is pure over a `ComputeEnvironment`.
"""

import pytest

from app.ml.compute import _notes
from app.ml.compute_plan import CPU_BATCH, MAX_BATCH, plan_for
from app.ml.compute_providers import providers
from app.schemas import ComputeDevice, ComputeEnvironment, ComputeFramework


def env(*, torch_devices=(), tf_devices=(), torch_installed=True, tf_installed=True):
    return ComputeEnvironment(
        machine="arm64",
        frameworks=[
            ComputeFramework(
                name="torch",
                installed=torch_installed,
                version="2.10.0",
                devices=list(torch_devices),
                accelerated=bool(torch_devices),
            ),
            ComputeFramework(
                name="tensorflow",
                installed=tf_installed,
                version="2.21.0",
                devices=list(tf_devices),
                accelerated=bool(tf_devices),
            ),
        ],
    )


def cuda(memory_mb=24576, capability="8.6"):
    return ComputeDevice(
        kind="cuda", index=0, name="NVIDIA RTX 4090", total_memory_mb=memory_mb, capability=capability
    )


def mps(memory_mb=12124):
    return ComputeDevice(kind="mps", index=0, name="Apple arm64 GPU", total_memory_mb=memory_mb)


# --- the disagreement note ----------------------------------------------------


def test_it_says_when_two_frameworks_see_different_hardware():
    """The finding this module exists for.

    Apple silicon without `tensorflow-metal` gives torch a GPU and TensorFlow a
    CPU. That is invisible until a Keras run takes twenty times longer than the
    torch run before it, and nobody guesses the cause.
    """
    notes = _notes(env(torch_devices=[mps()]))
    disagreement = next((note for note in notes if "but" in note), None)

    assert disagreement is not None
    assert "PyTorch sees a GPU" in disagreement
    assert "TensorFlow" in disagreement
    # The note is only useful if it names the fix.
    assert "tensorflow-metal" in disagreement


def test_no_disagreement_note_when_both_agree():
    for environment in (env(torch_devices=[cuda()], tf_devices=[cuda()]), env()):
        assert not any("but" in note and "sees a GPU" in note for note in _notes(environment))


def test_a_cpu_only_machine_is_told_plainly():
    notes = _notes(env())
    assert any("No GPU detected" in note for note in notes)


# --- device resolution --------------------------------------------------------


def test_a_keras_family_follows_tensorflow_not_torch():
    """The mapping is what makes a per-framework probe useful: a Keras run on a
    machine where only torch has a GPU is a CPU run, and both halves are needed
    to say so."""
    plan = plan_for(
        task_type="classification",
        model_family="keras_classification",
        environment=env(torch_devices=[mps()]),
    )
    assert plan.framework == "tensorflow"
    assert plan.device == "cpu"
    assert plan.accelerated is False
    assert any("CPU" in warning for warning in plan.warnings)


def test_a_torch_family_uses_the_gpu_torch_can_see():
    plan = plan_for(
        task_type="llm_finetune",
        model_family="llm_sft",
        environment=env(torch_devices=[mps()]),
    )
    assert plan.framework == "torch"
    assert plan.device == "mps"
    assert plan.accelerated is True


def test_an_explicit_device_is_honoured_even_when_the_probe_disagrees():
    """The user may know something the probe cannot see. Overriding their choice
    silently is worse than letting a clear error happen."""
    plan = plan_for(
        task_type="classification",
        model_family="yolo",
        requested_device="cuda",
        environment=env(torch_devices=[mps()]),
    )
    assert plan.device == "cuda"
    assert any("does not report it" in warning for warning in plan.warnings)


def test_a_missing_framework_is_a_warning_not_a_crash():
    plan = plan_for(
        task_type="classification",
        model_family="keras_classification",
        environment=env(tf_installed=False),
    )
    assert plan.device == "cpu"
    assert any("Install it" in warning for warning in plan.warnings)


# --- batch size ---------------------------------------------------------------


def test_batch_size_scales_with_memory():
    small = plan_for(task_type="t", model_family="yolo", environment=env(torch_devices=[mps(4096)]))
    large = plan_for(task_type="t", model_family="yolo", environment=env(torch_devices=[cuda(49152)]))
    assert large.batch_size > small.batch_size


def test_batch_size_is_capped_so_the_estimate_is_not_extrapolated_absurdly():
    """The per-sample footprint is a rough figure; trusting it out to a batch of
    500 would be treating an estimate as a measurement."""
    plan = plan_for(
        task_type="t", model_family="yolo", environment=env(torch_devices=[cuda(1_000_000)])
    )
    assert plan.batch_size <= MAX_BATCH


def test_a_cpu_run_gets_a_small_batch():
    plan = plan_for(task_type="t", model_family="yolo", environment=env())
    assert plan.batch_size == CPU_BATCH


def test_unknown_memory_keeps_the_default_rather_than_guessing():
    """TensorFlow reports no VRAM. Deriving a batch size from a number nobody
    has is how a default becomes confidently wrong."""
    device = ComputeDevice(kind="cuda", index=0, name="GPU", total_memory_mb=None)
    plan = plan_for(task_type="t", model_family="yolo", environment=env(torch_devices=[device]))
    assert plan.batch_size == 8
    assert any("does not report device memory" in reason for reason in plan.reasons)


def test_an_explicit_batch_size_is_warned_about_never_lowered():
    """An override is a decision. Quietly halving it would make the form lie
    about what ran."""
    plan = plan_for(
        task_type="t",
        model_family="llm_sft",
        requested_batch_size=64,
        environment=env(torch_devices=[mps(8192)]),
    )
    assert plan.batch_size == 64
    assert any("may not fit" in warning for warning in plan.warnings)


def test_a_heavier_family_gets_a_smaller_batch_on_the_same_card():
    light = plan_for(task_type="t", model_family="nlp_text_classification", environment=env(torch_devices=[cuda()]))
    heavy = plan_for(task_type="t", model_family="llm_sft", environment=env(torch_devices=[cuda()]))
    assert heavy.batch_size < light.batch_size


# --- precision ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("capability", "expected"),
    [("8.6", "bf16"), ("9.0", "bf16"), ("7.5", "fp16"), ("6.1", "fp16")],
)
def test_precision_follows_compute_capability(capability, expected):
    """bf16 needs Ampere or newer; on older cards fp16 is the one that actually
    speeds anything up."""
    plan = plan_for(
        task_type="t", model_family="yolo", environment=env(torch_devices=[cuda(capability=capability)])
    )
    assert plan.precision == expected


def test_metal_gets_fp16_not_bf16():
    plan = plan_for(task_type="t", model_family="yolo", environment=env(torch_devices=[mps()]))
    assert plan.precision == "fp16"


def test_cpu_stays_fp32():
    """Reduced precision on a CPU is slower, not faster."""
    plan = plan_for(task_type="t", model_family="yolo", environment=env())
    assert plan.precision == "fp32"


def test_every_plan_explains_itself():
    """A number alone is not actionable: "batch size 4" versus "batch size 4 —
    8 GB shared, ~1.8 GB per sample" is the difference between a value a user
    can argue with and one they cannot."""
    plan = plan_for(task_type="t", model_family="llm_sft", environment=env(torch_devices=[mps()]))
    assert len(plan.reasons) >= 3
    assert all(reason.strip() for reason in plan.reasons)


# --- providers ----------------------------------------------------------------


def test_only_local_is_available_and_the_rest_say_what_they_need():
    """Unavailable providers are listed rather than hidden so the roadmap is
    visible, and `requirements` doubles as the implementation checklist — a
    provider whose requirements nobody can write down is a wish, not a plan."""
    listed = {provider.id: provider for provider in providers()}

    assert listed["local"].available is True
    for identifier in ("vast", "modal", "runpod"):
        provider = listed[identifier]
        assert provider.available is False
        assert provider.requirements, f"{identifier} declares no requirements"
        assert provider.summary
