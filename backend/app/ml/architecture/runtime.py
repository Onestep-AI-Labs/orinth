"""Register the studio's built-in custom layers so a saved graph model reloads.

A graph that uses SqueezeExcite, PatchEmbedding, RMSNorm, or any other block
helper saves those layers into the `.keras` file by name. Reloading looks the
name up in Keras's global registry, and the registry is populated by *running*
the decorated class definition. Training does that as a side effect of importing
`generated_model.py`; nothing did it in the serving process, so every MobileNetV3
and ViT graph trained fine and then failed testing and inference with
"Could not locate class 'SqueezeExcite'".

`generated_model.py` cannot be the fix. Custom-layer nodes hold user Python, and
`specs/phase-17-model-architecture-studio.md` is explicit that user code never
runs in the FastAPI process. These classes are different: they are
platform-authored reference implementations that ship in this repository, so
importing this module registers them with no user code involved.

The class bodies are the same strings `emit_keras` writes into the generated
file, executed here rather than copied. A second, hand-maintained definition
would be a second source of truth, and the two would drift the first time a
block was fixed — at which point a model saved by the trainer would no longer
match the class the server rebuilds it with.

Importing this module imports TensorFlow, which is why it is not imported by the
emitters: code generation and shape inference stay TensorFlow-free.
"""

from __future__ import annotations

from typing import Any

from app.ml.architecture import emit_keras, keras_helpers

# Shared primitives first only for readability — every class body refers to the
# others from inside a method, so Python resolves them out of the namespace at
# call time and the order does not actually bind.
_HELPER_SOURCES: tuple[str, ...] = (
    keras_helpers.ROPE_APPLY_HELPER,
    keras_helpers.CAUSAL_MASK_HELPER,
    emit_keras.RMS_NORM_HELPER,
    emit_keras.ROPE_HELPER,
    emit_keras.SWIGLU_HELPER,
    emit_keras.POSITIONAL_HELPER,
    emit_keras.GQA_HELPER,
    emit_keras.MOE_HELPER,
    emit_keras.TIED_HEAD_HELPER,
    emit_keras.SOFTCAP_HELPER,
    keras_helpers.GATED_FFN_HELPER,
    keras_helpers.SPARSE_MOE_HELPER,
    keras_helpers.FAMILY_ATTENTION_HELPER,
    keras_helpers.LATENT_ATTENTION_HELPER,
    keras_helpers.SQUEEZE_EXCITE_HELPER,
    keras_helpers.PATCH_EMBEDDING_HELPER,
    keras_helpers.LAYER_SCALE_HELPER,
)

_registered = False


def register_builtin_layers() -> dict[str, Any]:
    """Define and register every built-in helper layer; return them by name.

    Idempotent: `register_keras_serializable` overwrites its entry, and the
    module-level guard keeps repeat calls from re-executing the bodies. The
    returned mapping suits `load_model(custom_objects=...)` for a caller that
    would rather be explicit than rely on the global registry.
    """

    global _registered
    import tensorflow as tf  # noqa: PLC0415 - the point of this module's laziness

    namespace: dict[str, Any] = {"tf": tf}
    for source in _HELPER_SOURCES:
        exec(source, namespace)  # noqa: S102 - repository source, not user input
    _registered = True
    return {
        name: value
        for name, value in namespace.items()
        if isinstance(value, type) and issubclass(value, tf.keras.layers.Layer)
    }


def ensure_registered() -> None:
    """Register the built-in layers once, before a `.keras` file is loaded."""

    if not _registered:
        register_builtin_layers()
