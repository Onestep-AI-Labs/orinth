"""Phase 14 LLM fine-tuning: environment probe, catalog gating, command
generation, chat-template pipeline, truncation counting, minimum rows,
asset preflight, and adapter auto-registration."""

import json
import shutil as shutil_module
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.storage import Storage
from app.db.models import TrainingJob
from app.ml.llm.catalog import (
    LLM_ADVANCED_PARAMETERS,
    catalog_base_ids,
    llm_model_id,
    llm_training_options,
    local_llm_base_options,
)
from app.ml.model_registry import ModelRegistry
from app.schemas import ModelAssetPrepareRequest, TrainingJobCreate
from app.services.datasets import DatasetService
from app.services.training import TrainingService
from app.schemas import DatasetCreate, DatasetRecordCreate
from app.training.runners.llm_sft import (
    LLM_ADVANCED_KEYS,
    encode_messages,
    ensure_chat_template,
    filter_supported_kwargs,
    load_jsonl_records,
    prepare_examples,
    record_messages,
    require_min_train_records,
    resolve_train_valid_records,
    sample_prompts,
    template_token_ids,
)


class FakeTokenizer:
    """Deterministic chat-template stand-in: one token per word, two per message.

    Mirrors the real transformers 5.x contract: ``tokenize=True`` with
    ``return_dict=True`` returns a ``{"input_ids", "attention_mask"}`` mapping
    (not a bare list), which is what the runner relies on. With
    ``add_generation_prompt=True`` one extra token is appended — enough to
    exercise masking and truncation without loading a real tokenizer.
    """

    def __init__(self, chat_template: str | None = "{{ messages }}"):
        self.chat_template = chat_template

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, return_dict=False, **_
    ):
        count = sum(2 + len(str(m.get("content", "")).split()) for m in messages)
        if add_generation_prompt:
            count += 1
        if tokenize:
            ids = list(range(count))
            if return_dict:
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
            return ids
        return " ".join(str(m.get("content", "")) for m in messages)


class BatchEncodingTokenizer(FakeTokenizer):
    """Like ``FakeTokenizer`` but always returns a dict for ``tokenize=True``.

    Transformers 5.x returns a ``BatchEncoding`` from
    ``apply_chat_template(tokenize=True)`` regardless of ``return_dict``; calling
    ``list()`` on it yields the keys, not the ids. This stand-in makes that
    regression reproducible.
    """

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, return_dict=False, **_
    ):
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            **_,
        )


def make_training_service(settings: Settings) -> TrainingService:
    storage = Storage(settings)
    storage.ensure()
    registry = ModelRegistry(settings, storage)
    dataset_service = DatasetService(settings, storage)
    return TrainingService(settings, storage, registry, dataset_service)


def make_llm_dataset(service: TrainingService, records: int = 12) -> str:
    dataset = service.dataset_service.create_dataset(
        DatasetCreate(name="SFT Data", task_type="llm_finetune", format="instruction_jsonl", labels=[])
    )
    for index in range(records):
        service.dataset_service.create_record(
            dataset.id,
            DatasetRecordCreate(
                split="train", record={"instruction": f"Question {index}", "output": f"Answer {index}"}
            ),
        )
    return dataset.id


# ---------------------------------------------------------------------------
# environment probe
# ---------------------------------------------------------------------------


def test_environment_probe_shape():
    from app.ml.llm.environment import probe_llm_environment

    env = probe_llm_environment()
    assert env.device in {"cuda", "mps", "cpu"}
    assert env.recommended_backend in {"unsloth", "peft"}
    # Recommending unsloth requires both CUDA and an importable unsloth.
    if env.recommended_backend == "unsloth":
        assert env.device == "cuda" and env.unsloth_available
    assert isinstance(env.notes, list)


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


def test_llm_catalog_lists_five_hub_bases_with_memory_guidance():
    ids = catalog_base_ids()
    assert ids == ["qwen2_5_1_5b", "smollm2_1_7b", "qwen2_5_3b", "gemma3_1b", "gemma3_4b"]
    assert llm_model_id("qwen2_5_1_5b") == "Qwen/Qwen2.5-1.5B-Instruct"
    options = {option.id: option for option in llm_training_options()}
    for base_id in ids:
        option = options[base_id]
        assert option.family == "llm_sft"
        assert option.task_types == ["llm_finetune"]
        assert "GB" in option.description, "memory guidance belongs in the description"
        assert option.advanced_parameters == LLM_ADVANCED_PARAMETERS


def test_llm_catalog_gates_on_missing_extras(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.ml.llm.catalog.llm_extras_available", lambda: False)
    options = llm_training_options()
    assert all(not option.runnable for option in options)
    assert all("uv sync --extra llm" in option.description for option in options)

    monkeypatch.setattr("app.ml.llm.catalog.llm_extras_available", lambda: True)
    assert all(option.runnable for option in llm_training_options())


def test_option_gated_flags_are_consistent_with_helper():
    # Gating is surfaced live via the HF model-info API (more accurate than a
    # static flag); the static default and the helper must at least agree.
    from app.ml.llm.catalog import llm_gated_license

    options = {option.id: option for option in llm_training_options()}
    for option_id in ("qwen2_5_1_5b", "smollm2_1_7b", "gemma3_1b", "gemma3_4b"):
        assert options[option_id].defaults["gated_license"] == llm_gated_license(option_id)


def test_local_llm_base_options_list_registered_hf_models(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("app.ml.llm.catalog.llm_extras_available", lambda: True)
    service = make_training_service(settings)
    base_dir = service.storage.uploaded_models / "uploaded_llm_hf_abc"
    base_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text("{}", encoding="utf-8")
    service.registry.register_model(
        model_id="uploaded_llm_hf_abc",
        name="My Base",
        family="llm_hf",
        task_type="llm_finetune",
        paths={"model": base_dir},
        labels=[],
        source="uploaded",
    )
    options = local_llm_base_options(service.registry)
    assert [option.id for option in options] == ["llm_local_uploaded_llm_hf_abc"]
    assert options[0].runnable
    assert options[0].source == "local"

    listed = service.model_options("llm_finetune")
    assert any(option.id == "llm_local_uploaded_llm_hf_abc" for option in listed)
    # The runner receives the registered model's filesystem path, not a hub id.
    assert service._llm_model_ref("llm_local_uploaded_llm_hf_abc") == str(base_dir)


def test_llm_advanced_keys_match_catalog_specs():
    assert {spec.key for spec in LLM_ADVANCED_PARAMETERS} == LLM_ADVANCED_KEYS


def test_finetune_method_is_a_catalog_option():
    options = {option.id: option for option in llm_training_options()}
    spec = next(s for s in options["qwen2_5_1_5b"].advanced_parameters if s.key == "finetune_method")
    assert spec.type == "select"
    assert set(spec.options) == {"lora", "qlora", "full", "continued_pretrain"}
    assert spec.default == "lora"


def test_run_config_resolves_each_finetune_method():
    from app.training.runners.llm_sft import _RunConfig

    lora = _RunConfig({"finetune_method": "lora"}, "cuda")
    assert lora.use_peft and not lora.load_in_4bit and not lora.continued_pretrain

    qlora = _RunConfig({"finetune_method": "qlora"}, "cuda")
    assert qlora.use_peft and qlora.load_in_4bit

    # QLoRA off CUDA downgrades to plain LoRA rather than failing.
    qlora_mps = _RunConfig({"finetune_method": "qlora"}, "mps")
    assert qlora_mps.use_peft and not qlora_mps.load_in_4bit

    full = _RunConfig({"finetune_method": "full"}, "cuda")
    assert not full.use_peft

    cont = _RunConfig({"finetune_method": "continued_pretrain"}, "cuda")
    assert cont.use_peft and cont.continued_pretrain and cont.train_on_full_sequence

    unknown = _RunConfig({"finetune_method": "bogus"}, "cuda")
    assert unknown.method == "lora"


def test_custom_hf_option_is_listed_and_flagged():
    options = {option.id: option for option in llm_training_options()}
    custom = options["llm_hf_custom"]
    assert custom.source == "huggingface"
    assert custom.defaults["custom_hf"] is True
    # No fixed model_id: the id is entered by the user and carried on base_model.
    assert "model_id" not in custom.defaults
    assert custom.advanced_parameters == LLM_ADVANCED_PARAMETERS


def test_trainable_format_note_names_gguf_and_tflite_as_non_trainable():
    from app.ml.llm.catalog import TRAINABLE_BASE_FAMILIES, TRAINABLE_FORMAT_NOTE

    assert TRAINABLE_BASE_FAMILIES == ["llm_hf"]
    assert "GGUF" in TRAINABLE_FORMAT_NOTE and "TFLite" in TRAINABLE_FORMAT_NOTE


# ---------------------------------------------------------------------------
# job creation gating
# ---------------------------------------------------------------------------


def test_create_job_without_extras_returns_install_hint(
    settings: Settings, db_session, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("app.ml.llm.catalog.llm_extras_available", lambda: False)
    monkeypatch.setattr("app.services.training.service.llm_extras_available", lambda: False)
    service = make_training_service(settings)
    payload = TrainingJobCreate(
        task_type="llm_finetune",
        model_family="llm_sft",
        model_option_id="qwen2_5_1_5b",
        epochs=1,
        batch_size=2,
        dataset_id="whatever",
    )
    with pytest.raises(ValueError, match="uv sync --extra llm"):
        service.create_job(db_session, payload)


def test_custom_hf_job_requires_a_model_id(
    settings: Settings, db_session, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("app.ml.llm.catalog.llm_extras_available", lambda: True)
    monkeypatch.setattr("app.services.training.service.llm_extras_available", lambda: True)
    service = make_training_service(settings)
    payload = TrainingJobCreate(
        task_type="llm_finetune",
        model_family="llm_sft",
        model_option_id="llm_hf_custom",
        epochs=1,
        batch_size=2,
        dataset_id="d1",
    )
    with pytest.raises(ValueError, match="Base model field"):
        service.create_job(db_session, payload)

    payload.base_model = "org/some-model"
    job = service.create_job(db_session, payload)
    assert job.parameters["base_model"] == "org/some-model"


def test_command_for_custom_hf_job_uses_the_typed_model_id(
    settings: Settings, tmp_path: Path
):
    service = make_training_service(settings)
    dataset_id = make_llm_dataset(service)
    job = TrainingJob(
        id="custom0001",
        model_family="llm_sft",
        status="queued",
        parameters={
            "task_type": "llm_finetune",
            "model_option_id": "llm_hf_custom",
            "base_model": "org/typed-model",
            "dataset_id": dataset_id,
            "epochs": 1,
            "batch_size": 2,
            "hyperparameters": {},
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    command = service._command_for_job(job, run_dir)
    assert command[command.index("--model-ref") + 1] == "org/typed-model"
    assert service._llm_model_ref("llm_hf_custom", "org/typed-model") == "org/typed-model"


def test_custom_hf_prepare_uses_the_request_model_ref(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    service = make_training_service(settings)

    class HugeDisk:
        free = 10**15

    monkeypatch.setattr("huggingface_hub.model_info", _fail_model_info)
    monkeypatch.setattr(shutil_module, "disk_usage", lambda _path: HugeDisk)
    downloaded: dict[str, str] = {}

    def fake_download(model_id, **kwargs):
        downloaded["model_id"] = model_id
        return "/tmp/snapshot"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    status = service.prepare_model_asset(
        ModelAssetPrepareRequest(option_id="llm_hf_custom", model_ref="org/typed-model")
    )
    assert status.status == "ready"
    assert downloaded["model_id"] == "org/typed-model"

    # Missing id is a clear, actionable miss rather than a download attempt.
    missing = service.prepare_model_asset(ModelAssetPrepareRequest(option_id="llm_hf_custom"))
    assert missing.status == "missing"


def test_create_job_with_extras_creates_an_llm_job(
    settings: Settings, db_session, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("app.ml.llm.catalog.llm_extras_available", lambda: True)
    monkeypatch.setattr("app.services.training.service.llm_extras_available", lambda: True)
    service = make_training_service(settings)
    payload = TrainingJobCreate(
        task_type="llm_finetune",
        model_family="llm_sft",
        model_option_id="qwen2_5_1_5b",
        epochs=1,
        batch_size=2,
        dataset_id="d1",
        learning_rate=0.0002,
    )
    job = service.create_job(db_session, payload)
    assert job.model_family == "llm_sft"
    assert job.parameters["task_type"] == "llm_finetune"


# ---------------------------------------------------------------------------
# command generation
# ---------------------------------------------------------------------------


def test_command_for_llm_job(settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    service = make_training_service(settings)
    dataset_id = make_llm_dataset(service)
    job = TrainingJob(
        id="job12345",
        model_family="llm_sft",
        status="queued",
        parameters={
            "task_type": "llm_finetune",
            "model_option_id": "qwen2_5_1_5b",
            "dataset_id": dataset_id,
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 0.0001,
            "hyperparameters": {"lora_r": 8, "max_seq_length": 1024},
        },
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    command = service._command_for_job(job, run_dir)
    assert "app.training.runners.llm_sft" in command
    ref_index = command.index("--model-ref") + 1
    assert command[ref_index] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert command[command.index("--epochs") + 1] == "2"
    assert command[command.index("--batch-size") + 1] == "4"
    advanced = json.loads(command[command.index("--advanced") + 1])
    assert advanced == {"lora_r": 8, "max_seq_length": 1024}
    # The dataset root must contain the phase-10 canonical train jsonl.
    dataset_root = Path(command[command.index("--dataset-root") + 1])
    assert (dataset_root / "train" / "data.jsonl").exists()


# ---------------------------------------------------------------------------
# data pipeline
# ---------------------------------------------------------------------------


def test_record_messages_templates_both_formats():
    instruction = record_messages({"instruction": "Sum 2+2", "input": "context", "output": "4"})
    assert instruction == [
        {"role": "user", "content": "Sum 2+2\n\ncontext"},
        {"role": "assistant", "content": "4"},
    ]
    chat = record_messages(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    )
    assert [m["role"] for m in chat] == ["user", "assistant"]
    assert record_messages({"instruction": "no output"}) is None


def test_template_token_ids_extracts_input_ids_from_batchencoding():
    # Regression: transformers 5.x returns a dict from apply_chat_template;
    # list() on it yields the keys (length 2) and collapsed every example.
    tokenizer = BatchEncodingTokenizer()
    messages = [
        {"role": "user", "content": "one two three"},
        {"role": "assistant", "content": "four five"},
    ]
    ids = template_token_ids(tokenizer, messages, add_generation_prompt=False)
    assert len(ids) > 2
    assert all(isinstance(token, int) for token in ids)


def test_encode_messages_survives_dict_returning_tokenizer():
    tokenizer = BatchEncodingTokenizer()
    messages = [
        {"role": "user", "content": "one two three"},
        {"role": "assistant", "content": "four five"},
    ]
    encoded = encode_messages(tokenizer, messages, max_seq_length=2048, mask_prompt=True)
    assert encoded is not None
    assert any(label != -100 for label in encoded["labels"])


def test_encode_messages_masks_prompt_tokens_by_default():
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "one two three"},
        {"role": "assistant", "content": "four five"},
    ]
    encoded = encode_messages(tokenizer, messages, max_seq_length=2048, mask_prompt=True)
    # Prompt = user message (2 + 3 tokens) + generation prompt (1) = 6 masked.
    assert encoded["labels"][:6] == [-100] * 6
    assert all(label != -100 for label in encoded["labels"][6:])
    assert encoded["truncated"] is False

    unmasked = encode_messages(tokenizer, messages, max_seq_length=2048, mask_prompt=False)
    assert all(label != -100 for label in unmasked["labels"])


def test_prepare_examples_counts_truncation():
    tokenizer = FakeTokenizer()
    records = [
        {"instruction": "short", "output": "ok"},
        {"instruction": "long " + "word " * 50, "output": "ok"},
    ]
    examples, stats = prepare_examples(tokenizer, records, max_seq_length=16, mask_prompt=False)
    assert stats["truncated"] == 1
    assert len(examples) == 2
    assert all(len(example["input_ids"]) <= 16 for example in examples)


def test_fully_masked_examples_are_dropped_not_trained():
    tokenizer = FakeTokenizer()
    # Truncation to 4 tokens removes the whole assistant span; a fully masked
    # example contributes nothing and must be dropped, not silently kept.
    records = [{"instruction": "one two three four five six", "output": "seven"}]
    examples, stats = prepare_examples(tokenizer, records, max_seq_length=4, mask_prompt=True)
    assert examples == []
    assert stats["skipped"] == 1


def test_default_chat_template_applied_when_tokenizer_lacks_one():
    bare = FakeTokenizer(chat_template=None)
    note = ensure_chat_template(bare)
    assert note and "ChatML" in note
    assert bare.chat_template

    owned = FakeTokenizer(chat_template="{% custom %}")
    assert ensure_chat_template(owned) is None
    assert owned.chat_template == "{% custom %}"


def test_minimum_row_rejection_names_the_split_flow():
    with pytest.raises(ValueError, match="Splits flow"):
        require_min_train_records(3)
    require_min_train_records(10)


def _write_split(root: Path, split: str, records: list[dict]) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "data.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def test_unsplit_dataset_falls_back_to_unassigned_with_a_holdout(tmp_path: Path):
    # All records in the `unassigned` inbox (recipe-commit / fresh-upload shape):
    # training still works instead of failing on an empty train split.
    records = [{"instruction": f"Q{i}", "output": f"A{i}"} for i in range(15)]
    _write_split(tmp_path, "unassigned", records)
    train, valid, source = resolve_train_valid_records(tmp_path, seed=42)
    assert source == "unassigned"
    assert len(train) >= 10
    assert len(valid) >= 2
    assert len(train) + len(valid) == 15
    # No record leaks across the train/valid boundary.
    train_keys = {json.dumps(record, sort_keys=True) for record in train}
    valid_keys = {json.dumps(record, sort_keys=True) for record in valid}
    assert train_keys.isdisjoint(valid_keys)


def test_real_train_split_is_preferred_over_the_inbox(tmp_path: Path):
    _write_split(tmp_path, "train", [{"instruction": f"T{i}", "output": "a"} for i in range(12)])
    _write_split(tmp_path, "valid", [{"instruction": "V", "output": "b"}])
    _write_split(tmp_path, "unassigned", [{"instruction": "U", "output": "c"}])
    train, valid, source = resolve_train_valid_records(tmp_path)
    assert source == "split"
    assert len(train) == 12
    # A real valid split is kept as-is; no holdout is carved from train.
    assert len(valid) == 1


def test_tiny_unassigned_pool_skips_the_holdout(tmp_path: Path):
    # Only 11 records: carving a holdout would drop train below the minimum, so
    # the whole pool trains and eval is simply skipped.
    _write_split(tmp_path, "unassigned", [{"instruction": f"Q{i}", "output": "a"} for i in range(11)])
    train, valid, source = resolve_train_valid_records(tmp_path)
    assert source == "unassigned"
    assert len(train) == 11
    assert valid == []


def test_load_jsonl_records_skips_malformed_lines(tmp_path: Path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    (split_dir / "data.jsonl").write_text(
        '{"instruction": "a", "output": "b"}\nnot json\n\n{"messages": []}\n', encoding="utf-8"
    )
    records = load_jsonl_records(tmp_path, "train")
    assert len(records) == 2


def test_sample_prompts_pull_reference_answers():
    prompts = sample_prompts(
        [
            {"instruction": "Q1", "output": "A1"},
            {"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]},
        ]
    )
    assert [item["prompt"] for item in prompts] == ["Q1", "Q2"]
    assert [item["reference"] for item in prompts] == ["A1", "A2"]


def test_filter_supported_kwargs_drops_unknown_names():
    def target(alpha: int = 1, beta: int = 2):
        return alpha + beta

    filtered = filter_supported_kwargs(target, {"alpha": 5, "gamma": 9})
    assert filtered == {"alpha": 5}


# ---------------------------------------------------------------------------
# model details (Hugging Face API)
# ---------------------------------------------------------------------------


class _Sibling:
    def __init__(self, size):
        self.size = size


class _Info:
    def __init__(self, sizes, gated=False, downloads=None, likes=None, library=None):
        self.siblings = [_Sibling(s) for s in sizes]
        self.gated = gated
        self.downloads = downloads
        self.likes = likes
        self.library_name = library


def test_model_details_resolves_catalog_id_and_reports_hub_size(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    service = make_training_service(settings)

    captured = {}

    def fake_model_info(ref, **kwargs):
        captured["ref"] = ref
        return _Info([1000, 2000, 500], downloads=12345, likes=67, library="transformers")

    monkeypatch.setattr("huggingface_hub.model_info", fake_model_info)
    info = service.llm_model_details("qwen2_5_1_5b")
    # The catalog id is resolved to its real hub id before the API call.
    assert captured["ref"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert info.exists and not info.gated
    assert info.size_bytes == 3500
    assert info.file_count == 3
    assert info.downloads == 12345


def test_model_details_flags_gated_and_missing_by_exception_type(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    service = make_training_service(settings)

    class GatedRepoError(Exception):
        pass

    class RepositoryNotFoundError(Exception):
        pass

    def gated(ref, **kwargs):
        raise GatedRepoError("Cannot access gated repo — its own message mentions gated")

    monkeypatch.setattr("huggingface_hub.model_info", gated)
    gated_info = service.llm_model_details("google/gemma-3-1b-it")
    assert gated_info.gated and gated_info.exists
    assert "gated" in (gated_info.error or "").lower()

    def missing(ref, **kwargs):
        # A 404's message mentions "gated/private" generically; type is what counts.
        raise RepositoryNotFoundError("404 ... private or gated repo ...")

    monkeypatch.setattr("huggingface_hub.model_info", missing)
    missing_info = service.llm_model_details("nope/not-real")
    assert missing_info.exists is False
    assert missing_info.gated is False
    assert "not found" in (missing_info.error or "").lower()


def test_model_details_for_local_base_reports_disk_size(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("app.ml.llm.catalog.llm_extras_available", lambda: True)
    service = make_training_service(settings)
    base_dir = service.storage.uploaded_models / "uploaded_llm_hf_xyz"
    base_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text('{"a": 1}', encoding="utf-8")
    (base_dir / "model.safetensors").write_text("weightsweights", encoding="utf-8")
    service.registry.register_model(
        model_id="uploaded_llm_hf_xyz",
        name="Local Base",
        family="llm_hf",
        task_type="llm_finetune",
        paths={"model": base_dir},
        labels=[],
        source="uploaded",
    )
    info = service.llm_model_details("llm_local_uploaded_llm_hf_xyz")
    assert info.exists
    assert info.library == "local"
    assert info.size_bytes and info.size_bytes > 0
    assert info.file_count == 2


# ---------------------------------------------------------------------------
# asset preparation
# ---------------------------------------------------------------------------


def _fail_model_info(*args, **kwargs):
    raise RuntimeError("offline")


def test_disk_preflight_fails_before_download(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
):
    service = make_training_service(settings)
    monkeypatch.setattr("huggingface_hub.model_info", _fail_model_info)

    class TinyDisk:
        free = 1024**3  # 1 GB free vs a multi-GB base

    monkeypatch.setattr(shutil_module, "disk_usage", lambda _path: TinyDisk)
    status = service.prepare_model_asset(ModelAssetPrepareRequest(option_id="qwen2_5_1_5b"))
    assert status.status == "failed"
    assert "GB" in (status.message or "")
    assert "free" in (status.message or "")


def test_gated_403_names_the_model_page(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    service = make_training_service(settings)
    monkeypatch.setattr("huggingface_hub.model_info", _fail_model_info)

    class HugeDisk:
        free = 10**15

    monkeypatch.setattr(shutil_module, "disk_usage", lambda _path: HugeDisk)

    def gated_download(*args, **kwargs):
        raise RuntimeError("403 Client Error: cannot access gated repo")

    monkeypatch.setattr("huggingface_hub.snapshot_download", gated_download)
    status = service.prepare_model_asset(ModelAssetPrepareRequest(option_id="gemma3_1b"))
    assert status.status == "gated"
    assert "https://huggingface.co/google/gemma-3-1b-it" in (status.message or "")


# ---------------------------------------------------------------------------
# auto-registration
# ---------------------------------------------------------------------------


def test_completed_llm_run_registers_an_adapter_model(settings: Settings, tmp_path: Path):
    service = make_training_service(settings)
    run_dir = tmp_path / "run"
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_text("stub", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"val_loss": 1.2}), encoding="utf-8")
    (run_dir / "sample_generations.json").write_text("[]", encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps({"artifact_type": "llm_adapter", "backend": "peft"}), encoding="utf-8"
    )
    job = TrainingJob(
        id="deadbeef1234",
        model_family="llm_sft",
        status="completed",
        parameters={
            "task_type": "llm_finetune",
            "model_option_id": "qwen2_5_1_5b",
            "model_name": "My Adapter",
        },
    )
    model_id = service._register_training_model(job, run_dir, adapter, {"val_loss": 1.2})
    info = next(model for model in service.registry.list_models() if model.id == model_id)
    assert info.family == "llm_adapter"
    assert info.source == "trained"
    assert info.task_type == "llm_finetune"
    assert info.base_model_id == "qwen2_5_1_5b"
    assert info.format == "safetensors"
    # ModelInfo falls back to DEFAULT_LABELS for empty lists; the stored
    # registry entry is what must carry no class labels.
    stored = json.loads(service.storage.registry_file.read_text(encoding="utf-8"))
    assert stored[model_id]["labels"] == []
    assert info.artifacts.get("backend") == "peft"
    model_path = Path(info.paths["model"])
    assert (model_path / "adapter_config.json").exists()


def test_local_base_registration_strips_the_option_prefix(settings: Settings, tmp_path: Path):
    service = make_training_service(settings)
    run_dir = tmp_path / "run"
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    job = TrainingJob(
        id="feedface5678",
        model_family="llm_sft",
        status="completed",
        parameters={
            "task_type": "llm_finetune",
            "model_option_id": "llm_local_uploaded_llm_hf_abc",
        },
    )
    model_id = service._register_training_model(job, run_dir, adapter, {})
    info = next(model for model in service.registry.list_models() if model.id == model_id)
    assert info.base_model_id == "uploaded_llm_hf_abc"


def test_full_finetune_registers_as_a_standalone_hf_model(settings: Settings, tmp_path: Path):
    # A full fine-tune saves a `model/` dir (not `adapter/`) and registers as a
    # standalone llm_hf model that phase 15 serves without a merge.
    service = make_training_service(settings)
    run_dir = tmp_path / "run"
    full_model = run_dir / "model"
    full_model.mkdir(parents=True)
    (full_model / "config.json").write_text("{}", encoding="utf-8")
    (full_model / "model.safetensors").write_text("weights", encoding="utf-8")
    job = TrainingJob(
        id="fu11m0de1",
        model_family="llm_sft",
        status="completed",
        parameters={
            "task_type": "llm_finetune",
            "model_option_id": "qwen2_5_1_5b",
            "hyperparameters": {"finetune_method": "full"},
        },
    )
    model_id = service._register_training_model(job, run_dir, full_model, {})
    info = next(model for model in service.registry.list_models() if model.id == model_id)
    assert info.family == "llm_hf"
    assert Path(info.paths["model"]).name == "model"


def test_ensure_sentencepiece_vocab_copies_and_noops():
    import tempfile

    from app.ml.llm.gguf_tools import ensure_sentencepiece_vocab

    # SentencePiece base → copies tokenizer.model into the output.
    base = Path(tempfile.mkdtemp())
    (base / "tokenizer.model").write_bytes(b"spm")
    out = Path(tempfile.mkdtemp())
    (out / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert ensure_sentencepiece_vocab(out, str(base)) is True
    assert (out / "tokenizer.model").exists()
    # Already present → no-op; BPE base (no sentencepiece) → no-op.
    assert ensure_sentencepiece_vocab(out, str(base)) is False
    bpe = Path(tempfile.mkdtemp())
    (bpe / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert ensure_sentencepiece_vocab(Path(tempfile.mkdtemp()), str(bpe)) is False
