"""Phase 13 advanced training settings.

Covers the catalog-driven mechanism end to end at the seams a stale frontend or
a hand-edited request actually hit: catalog serialization, each runner's
allowlist mapping, ignore-unknown behaviour, and precision downgrade.
"""

from app.ml.training_catalog import training_model_options
from app.schemas import AdvancedParameterSpec
from app.training.runners import advanced as advanced_helper
from app.training.runners.advanced import parse_advanced, partition, resolve_precision


def _options_by_id() -> dict[str, object]:
    return {option.id: option for option in training_model_options()}


def test_all_four_families_serialize_non_empty_advanced_parameters():
    options = _options_by_id()
    representatives = {
        "yolo": "ultralytics_yolo11_detect",
        "keras": "keras_efficientnet_b0",
        "hf": "hf_distilbert_text_classifier",
        "sklearn": "nlp_tfidf_classifier",
    }
    for family, option_id in representatives.items():
        option = options[option_id]
        assert option.advanced_parameters, f"{family} option {option_id} has no advanced params"
        for spec in option.advanced_parameters:
            assert isinstance(spec, AdvancedParameterSpec)
            assert spec.key and spec.label and spec.group


def test_advanced_specs_group_into_the_documented_headers():
    allowed_groups = {
        "Method",
        "LoRA",
        "Quantization",
        "Sequence",
        "Optimization",
        "Augmentation",
        "Regularization",
        "Generation",
        "Runtime",
    }
    for option in training_model_options():
        for spec in option.advanced_parameters:
            assert spec.group in allowed_groups
        # A select must offer options; a number must not masquerade as one.
        for spec in option.advanced_parameters:
            if spec.type in {"select", "multiselect"}:
                assert spec.options, f"{spec.key} is a select with no options"


def test_gated_options_stay_bare():
    options = _options_by_id()
    # SAM is a catalog-only gated entry; it must not advertise runnable knobs.
    assert options["ultralytics_sam3"].advanced_parameters == []


def test_hf_summarization_shares_the_same_advanced_set_as_classification():
    options = _options_by_id()
    classification = {spec.key for spec in options["hf_distilbert_text_classifier"].advanced_parameters}
    summarization = {spec.key for spec in options["hf_bart_summarizer"].advanced_parameters}
    assert classification == summarization
    assert "precision" in classification and "warmup_ratio" in classification


def test_partition_keeps_known_keys_and_reports_unknown():
    accepted, ignored = partition(
        {"lrf": 0.2, "mosaic": 0.7, "totally_made_up": 5}, {"lrf", "mosaic"}
    )
    assert accepted == {"lrf": 0.2, "mosaic": 0.7}
    assert ignored == ["totally_made_up"]


def test_partition_drops_none_but_does_not_call_it_unknown():
    # An empty multiselect submits None, meaning "runner default", not a value.
    accepted, ignored = partition({"seed": None}, {"seed"})
    assert accepted == {}
    assert ignored == []


def test_parse_advanced_tolerates_garbage():
    assert parse_advanced(None) == {}
    assert parse_advanced("") == {}
    assert parse_advanced("not json") == {}
    assert parse_advanced("[1, 2]") == {}
    assert parse_advanced('{"a": 1}') == {"a": 1}


def test_yolo_runner_maps_advanced_onto_train_args_and_ignores_unknown():
    from app.training.runners.yolo_train import build_advanced_train_args

    accepted, ignored = build_advanced_train_args(
        '{"lrf": 0.25, "cos_lr": true, "warmup_epochs": 3, "nonsense": 1}'
    )
    assert accepted == {"lrf": 0.25, "cos_lr": True, "warmup_epochs": 3}
    assert ignored == ["nonsense"]


def test_keras_runner_allowlist_ignores_unknown():
    from app.training.runners.keras_classification_train import select_keras_advanced

    accepted, ignored = select_keras_advanced('{"dropout": 0.5, "made_up": 2}')
    assert accepted == {"dropout": 0.5}
    assert ignored == ["made_up"]


def test_sklearn_runner_resolves_tfidf_and_classifier_kwargs():
    from app.training.runners.nlp.baseline import parse_ngram_range, tfidf_classifier_settings

    assert parse_ngram_range("1,3") == (1, 3)
    assert parse_ngram_range("bad") == (1, 2)
    assert parse_ngram_range("3,1") == (1, 2)

    vectorizer_kwargs, classifier_kwargs, ignored = tfidf_classifier_settings(
        {"C": 2.5, "max_iter": 400, "tfidf_ngram_range": "1,1", "class_weight": "balanced", "x": 1},
        epochs=5,
        learning_rate=1.0,
    )
    assert vectorizer_kwargs["ngram_range"] == (1, 1)
    assert classifier_kwargs["C"] == 2.5
    assert classifier_kwargs["max_iter"] == 400
    assert classifier_kwargs["class_weight"] == "balanced"
    assert ignored == ["x"]


def test_sklearn_settings_fall_back_to_basic_fields():
    from app.training.runners.nlp.baseline import tfidf_classifier_settings

    vectorizer_kwargs, classifier_kwargs, _ = tfidf_classifier_settings({}, epochs=5, learning_rate=1.0)
    assert "max_features" not in vectorizer_kwargs
    assert classifier_kwargs["max_iter"] == 500  # max(epochs, 1) * 100
    assert classifier_kwargs["C"] == 1.0
    assert classifier_kwargs["class_weight"] is None


def test_precision_downgrades_off_cuda_and_passes_through_on_cuda():
    assert resolve_precision("bf16", cuda=False, mps=True)[0] == "fp32"
    assert resolve_precision("fp16", cuda=False, mps=False)[0] == "fp32"
    assert resolve_precision("bf16", cuda=True, mps=False) == ("bf16", None)
    assert resolve_precision("fp32", cuda=False, mps=False) == ("fp32", None)
    # An unknown request never crashes a run; it downgrades with a note.
    effective, note = resolve_precision("int4", cuda=True, mps=False)
    assert effective == "fp32" and note


def test_precision_downgrade_notes_name_the_device():
    _, mps_note = resolve_precision("bf16", cuda=False, mps=True)
    _, cpu_note = resolve_precision("bf16", cuda=False, mps=False)
    assert "MPS" in mps_note
    assert "CPU" in cpu_note


def test_allowed_keys_are_derived_from_catalog_specs():
    # The runner allowlist and the catalog fields cannot drift: they are the
    # same set.
    options = _options_by_id()
    yolo_keys = {spec.key for spec in options["yolo_local"].advanced_parameters}
    from app.training.runners.yolo_train import YOLO_ADVANCED_KEYS

    assert yolo_keys == YOLO_ADVANCED_KEYS
    assert advanced_helper.allowed_keys(options["yolo_local"].advanced_parameters) == YOLO_ADVANCED_KEYS
