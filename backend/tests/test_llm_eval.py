"""Phase 15 LLM evaluation: progress parsing, per-example scoring math, the
GGUF rejection guard, and that LLM datasets surface on the testing endpoint."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.ml.model_registry import ModelRegistry
from app.services.datasets import DatasetService
from app.services.evaluation.service import EvaluationService, _parse_eval_progress
from app.training.runners.llm_eval import score_example


def test_parse_eval_progress_maps_to_scoring_band():
    percent, processed, total = _parse_eval_progress("eval 5/10")
    assert (processed, total) == (5, 10)
    assert 20 < percent < 95  # scoring occupies the 20–95% band
    assert _parse_eval_progress("Loading model: Qwen") == (None, 0, 0)


class _FixedLogitsModel:
    """Torch stand-in returning preset logits, so scoring math is deterministic."""

    def __init__(self, logits):
        import torch  # noqa: PLC0415

        self._logits = torch.tensor([logits], dtype=torch.float32)

    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        return SimpleNamespace(logits=self._logits)


def test_score_example_perfect_prediction():
    import torch  # noqa: PLC0415

    # labels mask the first two positions; positions 1 and 2 predict tokens 3 and 4.
    example = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "labels": [-100, -100, 3, 4],
    }
    logits = [
        [0.0, 0, 0, 0, 0],  # pos 0 (unused: shift label -100)
        [0.0, 0, 0, 10, 0],  # pos 1 predicts token 3
        [0.0, 0, 0, 0, 10],  # pos 2 predicts token 4
        [0.0, 0, 0, 0, 0],  # pos 3 dropped by the shift
    ]
    summed_loss, scored, correct = score_example(_FixedLogitsModel(logits), example, torch.device("cpu"))
    assert scored == 2
    assert correct == 2
    assert summed_loss / scored < 0.01  # confident + correct → near-zero loss


def test_score_example_wrong_prediction_counts_zero_correct():
    import torch  # noqa: PLC0415

    example = {
        "input_ids": [1, 2, 3, 4],
        "attention_mask": [1, 1, 1, 1],
        "labels": [-100, -100, 3, 4],
    }
    logits = [
        [0.0, 0, 0, 0, 0],
        [10.0, 0, 0, 0, 0],  # predicts token 0, not 3
        [10.0, 0, 0, 0, 0],  # predicts token 0, not 4
        [0.0, 0, 0, 0, 0],
    ]
    summed_loss, scored, correct = score_example(_FixedLogitsModel(logits), example, torch.device("cpu"))
    assert scored == 2
    assert correct == 0
    assert summed_loss > 0


def test_evaluate_llm_rejects_gguf(settings, storage):
    registry = ModelRegistry(settings, storage)
    dataset_service = DatasetService(settings, storage)
    service = EvaluationService(settings, storage, registry, dataset_service)
    gguf_dir = storage.trained_models / "g"
    gguf_dir.mkdir(parents=True)
    (gguf_dir / "model.gguf").write_bytes(b"GGUF")
    registry.register_model(
        model_id="g1",
        name="GGUF",
        family="llm_gguf",
        task_type="llm_finetune",
        paths={"model": gguf_dir / "model.gguf"},
        labels=[],
        format="gguf",
    )
    spec = registry.get_spec("g1")
    job = SimpleNamespace(id="job1", dataset_key="dataset:d:test", limit=None)
    with pytest.raises(ValueError, match="GGUF models are tested"):
        service._evaluate_llm(None, job, datetime.now(UTC).replace(tzinfo=None), spec)


def test_llm_dataset_surfaces_on_testing_list(settings, storage):
    from app.schemas import DatasetCreate, DatasetRecordCreate

    dataset_service = DatasetService(settings, storage)
    registry = ModelRegistry(settings, storage)
    service = EvaluationService(settings, storage, registry, dataset_service)

    created = dataset_service.create_dataset(
        DatasetCreate(name="Chat eval set", task_type="llm_finetune", format="chat_jsonl", labels=[])
    )
    # Seed a couple of records into the test split so the split is non-empty.
    for index in range(2):
        dataset_service.create_record(
            created.id,
            DatasetRecordCreate(
                split="test",
                record={
                    "messages": [
                        {"role": "user", "content": f"q{index}"},
                        {"role": "assistant", "content": f"a{index}"},
                    ]
                },
            ),
        )
    datasets = service.list_datasets(task_type="llm_finetune")
    keys = [dataset.key for dataset in datasets]
    assert f"dataset:{created.id}:test" in keys
