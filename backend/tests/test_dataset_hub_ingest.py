"""The as-is Hub import: download a split unchanged, and stop there.

The network is stubbed at `_fetch_page` / `_split_rows` / `_download_asset` —
the boundary where datasets-server ends — and everything below it is real: the
staging layout, the detector, the plan, and the files the training runners load.

Two things are being proved, and the split between them is the point. Importing
lands the files and leaves a **draft**: a Hub dataset is under no obligation to
be shaped like something Orinth trains, and a download that worked must not
report itself as failed because a plan could not be named. Preparing is then a
separate request — the "Prepare with Orinth" button — and when it runs, the
files are in a shape detection already understands.
"""

import json

import pytest
from PIL import Image

from app.schemas import DatasetHubIngestRequest
from app.services.dataset_hub import DatasetHubService
from app.services.datasets import DatasetService
from app.services.datasets.prep.service import DatasetPrepService
from app.services.datasets.prep.staging import staging_root


def rows_and_features(rows, features):
    """A `_fetch_page` stand-in serving one fixed page and then nothing."""

    def fetch(hub_id, config, split, offset, length):
        del hub_id, config, split
        return rows[offset : offset + length], features

    return fetch


@pytest.fixture
def hub(settings, storage, monkeypatch):
    datasets = DatasetService(settings, storage)
    prep = DatasetPrepService(settings, datasets)
    service = DatasetHubService(settings, storage, datasets, prep)
    # `_configs_and_splits` and `_revision` are the two other calls that reach
    # the network before any row is read.
    monkeypatch.setattr(service, "_configs_and_splits", lambda *_a, **_k: (["default"], ["train"]))
    monkeypatch.setattr(service, "_revision", lambda *_a, **_k: "abc123")
    return service


def request(**overrides) -> DatasetHubIngestRequest:
    payload = {"project_id": "default", "hub_id": "acme/reviews", "max_rows": 60}
    payload.update(overrides)
    return DatasetHubIngestRequest(**payload)


def test_a_row_split_stages_as_jsonl_and_stops_at_downloaded(hub, monkeypatch):
    rows = [
        {"text": f"the service was {mood} and the plates arrived {speed}", "label": index % 2}
        for index, (mood, speed) in enumerate([("good", "fast"), ("bad", "late")] * 30)
    ]
    features = [
        {"name": "text", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "label", "type": {"_type": "ClassLabel", "names": ["good", "bad"]}},
    ]
    monkeypatch.setattr(hub, "_fetch_page", rows_and_features(rows, features))
    monkeypatch.setattr(hub, "_split_rows", lambda *_a, **_k: len(rows))

    summary = hub.ingest_hub(request())

    # `ClassLabel` integers are resolved to their names on the way to disk: a
    # label column full of `1` is not a taxonomy anything can read.
    staged = list(staging_root(hub.storage.datasets / summary.id).glob("*.jsonl"))
    assert len(staged) == 1
    written = [json.loads(line) for line in staged[0].read_text().splitlines()]
    assert {row["label"] for row in written} == {"good", "bad"}

    # Downloaded, not prepared. The files are in the workspace and nothing has
    # been decided about them.
    status = hub.prep.prep_status(summary.id)
    assert status.state == "draft"
    assert "Downloaded" in status.detail
    downloaded = hub.datasets.summary(summary.id)
    assert downloaded.origin == "imported_hf"
    assert downloaded.origin_ref == "acme/reviews@abc123"
    assert sum(split.item_count for split in downloaded.splits.values()) == 0

    # …and preparing it is the separate request the studio's button makes.
    hub.prep.run(summary.id)
    prepared = hub.datasets.summary(summary.id)
    assert prepared.task_type == "text_classification"
    assert sum(split.item_count for split in prepared.splits.values()) == len(rows)


def test_an_image_split_stages_as_one_folder_per_class(hub, monkeypatch, tmp_path):
    rows = [{"img": {"src": f"https://cdn.test/{index}.png"}, "label": index % 2} for index in range(20)]
    features = [
        {"name": "img", "type": {"_type": "Image"}},
        {"name": "label", "type": {"_type": "ClassLabel", "names": ["cat", "dog"]}},
    ]
    monkeypatch.setattr(hub, "_fetch_page", rows_and_features(rows, features))
    monkeypatch.setattr(hub, "_split_rows", lambda *_a, **_k: len(rows))

    def download(url, destination):
        del url
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), "white").save(destination)
        return True

    monkeypatch.setattr(hub, "_download_asset", download)

    summary = hub.ingest_hub(request(hub_id="acme/pets", max_rows=20))
    staging = staging_root(hub.storage.datasets / summary.id)

    # The folder tree *is* the label signal. Nothing else is written beside the
    # images: a `metadata.jsonl` sidecar made record detection — which runs
    # first — read a folder of pictures as a table of rows.
    assert sorted(entry.name for entry in staging.iterdir() if entry.is_dir()) == ["cat", "dog"]
    assert not list(staging.glob("*.jsonl"))
    assert hub.prep.prep_status(summary.id).state == "draft"

    hub.prep.run(summary.id)
    final = hub.datasets.summary(summary.id)
    assert final.task_type == "classification"
    assert sorted(final.labels) == ["cat", "dog"]
    assert sum(split.item_count for split in final.splits.values()) == 20


def test_an_empty_split_fails_the_dataset_rather_than_leaving_it_pending(hub, monkeypatch):
    monkeypatch.setattr(hub, "_fetch_page", rows_and_features([], []))
    monkeypatch.setattr(hub, "_split_rows", lambda *_a, **_k: 0)

    summary = hub.ingest_hub(request(hub_id="acme/empty"))
    status = hub.prep.prep_status(summary.id)

    assert status.state == "failed"
    assert "no rows" in (status.error or "").lower()


def test_the_download_reports_its_own_progress(hub, monkeypatch):
    rows = [{"text": f"row {index} with several words of prose", "label": index % 2} for index in range(40)]
    features = [
        {"name": "text", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "label", "type": {"_type": "ClassLabel", "names": ["a", "b"]}},
    ]
    monkeypatch.setattr(hub, "_fetch_page", rows_and_features(rows, features))
    monkeypatch.setattr(hub, "_split_rows", lambda *_a, **_k: len(rows))

    seen: list[tuple[str, float | None]] = []
    original = hub.prep._set_state

    def record(dataset_id, state, plan=None, **kwargs):
        if kwargs.get("step"):
            seen.append((kwargs["step"], kwargs.get("progress")))
        return original(dataset_id, state, plan, **kwargs)

    monkeypatch.setattr(hub.prep, "_set_state", record)
    hub.ingest_hub(request(hub_id="acme/progress"))

    assert seen[0][0] == "staging"
    # The download owns the first band and the agent is floored above it, so the
    # bar never travels backwards at the hand-off.
    fractions = [progress for _step, progress in seen if progress is not None]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0


def test_a_split_sorted_by_label_is_sampled_across_not_off_the_head(hub, monkeypatch):
    """The `stanfordnlp/imdb` failure, reproduced.

    imdb stores its train split sorted by label: rows 0-12,499 are `neg` and
    12,500-24,999 are `pos`. Taking a prefix of a perfectly balanced dataset
    therefore yields exactly one class, detection refuses to call a
    single-valued column a taxonomy, and the import reads as a failure on a
    dataset Orinth trains happily.
    """
    total = 400
    rows = [
        {"text": f"review number {index} with several words of prose in it",
         "label": 0 if index < total // 2 else 1}
        for index in range(total)
    ]
    features = [
        {"name": "text", "type": {"dtype": "string", "_type": "Value"}},
        {"name": "label", "type": {"_type": "ClassLabel", "names": ["neg", "pos"]}},
    ]
    monkeypatch.setattr(hub, "_fetch_page", rows_and_features(rows, features))
    monkeypatch.setattr(hub, "_split_rows", lambda *_a, **_k: total)

    summary = hub.ingest_hub(request(hub_id="acme/sorted", max_rows=40))
    staged = list(staging_root(hub.storage.datasets / summary.id).glob("*.jsonl"))
    written = [json.loads(line) for line in staged[0].read_text().splitlines()]

    assert len(written) == 40
    # Both classes present, and in the proportion the split actually holds.
    assert {row["label"] for row in written} == {"neg", "pos"}
    assert sum(row["label"] == "pos" for row in written) == 20

    hub.prep.run(summary.id)
    prepared = hub.datasets.summary(summary.id)
    assert prepared.task_type == "text_classification"
    assert sorted(prepared.labels) == ["neg", "pos"]
