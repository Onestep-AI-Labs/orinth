"""Detection rules, one synthetic tree per row of the phase-21 table.

Each test builds the directory a user would actually drop in and asserts what
Orinth concludes from it. The trees are deliberately minimal — detection reads
structure, so structure is all a fixture needs to carry.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from app.services.datasets.prep.detect import detect, detect_record_format


def image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(path)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# --- rows 1-3: YOLO -----------------------------------------------------------


def test_yolo_with_data_yaml_detects_object_detection(tmp_path: Path):
    image(tmp_path / "train" / "images" / "a.jpg")
    write(tmp_path / "train" / "labels" / "a.txt", "0 0.5 0.5 0.2 0.2\n")
    write(tmp_path / "data.yaml", "names: ['lesion', 'cyst']\n")

    found = detect(tmp_path)
    assert found.modality == "image"
    assert found.task_type == "object_detection"
    assert found.format == "yolo"
    assert found.candidate_labels == ["lesion", "cyst"]
    assert found.confidence >= 0.9
    assert any("data.yaml" in signal for signal in found.signals)


def test_polygon_label_lines_detect_segmentation(tmp_path: Path):
    """A 5-token line is a box; an odd, longer line is a polygon."""
    image(tmp_path / "train" / "images" / "a.jpg")
    write(
        tmp_path / "train" / "labels" / "a.txt",
        "0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4\n",
    )
    write(tmp_path / "data.yaml", "names: ['lesion']\n")

    found = detect(tmp_path)
    assert found.task_type == "segmentation"
    assert found.format == "yolo"


def test_yolo_without_data_yaml_is_low_confidence_and_needs_labels(tmp_path: Path):
    image(tmp_path / "train" / "images" / "a.jpg")
    write(tmp_path / "train" / "labels" / "a.txt", "0 0.5 0.5 0.2 0.2\n")

    found = detect(tmp_path)
    assert found.task_type == "object_detection"
    assert found.candidate_labels == []
    assert found.confidence < 0.7
    assert found.needs_input is not None


# --- rows 4-5: COCO -----------------------------------------------------------


def test_coco_with_masks_detects_segmentation(tmp_path: Path):
    image(tmp_path / "train" / "img" / "a.jpg")
    write(
        tmp_path / "train" / "img" / "_annotations.coco.json",
        json.dumps(
            {
                "categories": [{"id": 1, "name": "lesion"}],
                "annotations": [{"id": 1, "category_id": 1, "segmentation": [[1, 2, 3, 4]]}],
            }
        ),
    )
    found = detect(tmp_path)
    assert found.task_type == "segmentation"
    assert found.format == "coco"
    assert found.candidate_labels == ["lesion"]


def test_coco_with_boxes_only_detects_detection(tmp_path: Path):
    image(tmp_path / "img" / "a.jpg")
    write(
        tmp_path / "img" / "_annotations.coco.json",
        json.dumps(
            {
                "categories": [{"id": 1, "name": "lesion"}],
                "annotations": [{"id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}],
            }
        ),
    )
    found = detect(tmp_path)
    assert found.task_type == "object_detection"
    assert found.format == "coco"


def test_malformed_coco_sidecar_does_not_crash_detection(tmp_path: Path):
    image(tmp_path / "img" / "a.jpg")
    write(tmp_path / "img" / "_annotations.coco.json", "{ not json")
    found = detect(tmp_path)
    # Falls through to the image rules rather than raising.
    assert found.modality == "image"


# --- rows 6-7: image folders --------------------------------------------------


def test_per_class_image_folders_read_folder_names_as_labels(tmp_path: Path):
    for label in ("normal", "kista", "granuloma"):
        for index in range(2):
            image(tmp_path / label / f"{index}.jpg")

    found = detect(tmp_path)
    assert found.task_type == "classification"
    assert found.format == "image_folder"
    assert found.candidate_labels == ["granuloma", "kista", "normal"]
    assert any("3 folders" in signal for signal in found.signals)


def test_flat_unlabelled_images_ask_for_labels_instead_of_inventing_them(tmp_path: Path):
    """The case that used to train silently on zero items."""
    for index in range(5):
        image(tmp_path / f"{index}.jpg")

    found = detect(tmp_path)
    assert found.task_type == "classification"
    assert found.candidate_labels == []
    assert found.confidence < 0.5
    assert found.needs_input is not None
    assert "label" in found.needs_input.lower()


# --- rows 8-9: LLM records ----------------------------------------------------


def test_alpaca_jsonl_detects_instruction_finetuning(tmp_path: Path):
    write(
        tmp_path / "data.jsonl",
        "\n".join(
            json.dumps({"instruction": f"q{i}", "input": "", "output": f"a{i}"})
            for i in range(3)
        ),
    )
    found = detect(tmp_path)
    assert found.modality == "record"
    assert found.task_type == "llm_finetune"
    assert found.format == "instruction_jsonl"
    assert found.sample_rows


def test_messages_jsonl_detects_chat_finetuning(tmp_path: Path):
    write(
        tmp_path / "chat.jsonl",
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
    )
    found = detect(tmp_path)
    assert found.task_type == "llm_finetune"
    assert found.format == "chat_jsonl"


def test_sharegpt_conversations_detect_chat_finetuning(tmp_path: Path):
    write(
        tmp_path / "chat.jsonl",
        json.dumps({"conversations": [{"from": "human", "value": "hi"}]}),
    )
    found = detect(tmp_path)
    assert found.format == "chat_jsonl"


def test_blank_lines_and_bad_json_are_skipped_not_fatal(tmp_path: Path):
    write(
        tmp_path / "data.jsonl",
        '\n{"instruction": "q", "output": "a"}\n{ broken\n\n',
    )
    found = detect(tmp_path)
    assert found.format == "instruction_jsonl"


# --- rows 10-13: text and delimited files -------------------------------------


def test_csv_with_question_and_answer_detects_qa(tmp_path: Path):
    write(tmp_path / "qa.csv", "question,answer\nwhat?,this\nwho?,them\n")
    found = detect(tmp_path)
    assert found.task_type == "question_answering"
    assert found.candidate_labels == ["answer"]


def test_csv_with_text_and_summary_detects_summarization(tmp_path: Path):
    write(tmp_path / "s.csv", "text,summary\nlong body here,short\nanother body,brief\n")
    found = detect(tmp_path)
    assert found.task_type == "summarization"
    assert found.candidate_labels == ["summary"]


def test_csv_with_text_and_label_detects_text_classification(tmp_path: Path):
    write(
        tmp_path / "c.csv",
        "text,label\nlovely,positive\nawful,negative\ngreat,positive\n",
    )
    found = detect(tmp_path)
    assert found.task_type == "text_classification"
    assert found.candidate_labels == ["negative", "positive"]
    assert any("distinct values" in signal for signal in found.signals)


def test_a_high_cardinality_target_column_is_not_treated_as_classes(tmp_path: Path):
    """A unique value per row is free text, not a class taxonomy."""
    rows = "\n".join(f"body {i},summary text number {i}" for i in range(30))
    write(tmp_path / "c.csv", f"text,target\n{rows}\n")
    found = detect(tmp_path)
    assert found.task_type != "text_classification"


def test_txt_files_in_class_folders_detect_text_classification(tmp_path: Path):
    for label in ("pos", "neg"):
        for index in range(2):
            write(tmp_path / label / f"{index}.txt", "some words")
    found = detect(tmp_path)
    assert found.task_type == "text_classification"
    assert found.candidate_labels == ["neg", "pos"]


def test_a_flat_txt_corpus_detects_language_modeling(tmp_path: Path):
    for index in range(3):
        write(tmp_path / f"{index}.txt", "a paragraph of prose")
    found = detect(tmp_path)
    assert found.task_type == "language_modeling"


# --- rows 14-15: fallbacks ----------------------------------------------------


def test_an_unmappable_table_reports_table_modality_and_asks(tmp_path: Path):
    write(tmp_path / "t.csv", "alpha,beta,gamma\n1,2,3\n4,5,6\n")
    found = detect(tmp_path)
    assert found.modality == "table"
    assert found.task_type is None
    assert found.needs_input is not None
    assert found.columns == ["alpha", "beta", "gamma"]


def test_parquet_hands_off_to_the_tabular_studio(tmp_path: Path):
    (tmp_path / "t.parquet").write_bytes(b"PAR1notreallyparquet")
    found = detect(tmp_path)
    assert found.modality == "table"
    assert found.task_type is None


def test_an_empty_directory_is_reported_not_crashed(tmp_path: Path):
    found = detect(tmp_path)
    assert found.modality == "unknown"
    assert found.signals


def test_a_missing_directory_is_reported_not_crashed(tmp_path: Path):
    found = detect(tmp_path / "nope")
    assert found.modality == "unknown"


def test_unrecognized_files_fall_through_to_needs_input(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"\x00\x01")
    found = detect(tmp_path)
    assert found.task_type is None
    assert found.needs_input is not None


# --- shared record-shape sniffer ----------------------------------------------


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["instruction", "output"], "alpaca"),
        (["instruction", "input", "output"], "alpaca"),
        (["messages"], "messages"),
        (["conversations"], "sharegpt"),
        (["question", "answer"], "qa"),
        # `rajpurkar/squad`, the Hub's most-downloaded QA dataset, names its
        # column `answers`. Requiring the singular made it detect as nothing and
        # drop into manual mapping — see the alias test below.
        (["id", "title", "context", "question", "answers"], "qa"),
        (["query", "answer"], "qa"),
        (["alpha", "beta"], None),
    ],
)
def test_detect_record_format_matches_the_hub_importers_vocabulary(columns, expected):
    """`DatasetHubService._detect_format` now delegates here; same answers."""
    shape, _mapping = detect_record_format(columns)
    assert shape == expected


def test_the_qa_mapping_names_the_real_columns_not_the_canonical_roles():
    """The mapping is what the importer reads columns by, so it has to carry the
    dataset's own names — `answers`, not the role name `answer`."""
    shape, mapping = detect_record_format(["id", "title", "context", "question", "answers"])
    assert shape == "qa"
    assert mapping == {"question": "question", "answer": "answers", "context": "context"}


# --- prompt safety ------------------------------------------------------------


def test_sample_rows_are_truncated_so_one_huge_cell_cannot_blow_the_prompt(tmp_path: Path):
    from app.services.datasets.prep.detect import SAMPLE_CELL_CHARS, SAMPLE_ROW_LIMIT

    rows = "\n".join(
        json.dumps({"instruction": "q", "output": "x" * 5000}) for _ in range(50)
    )
    write(tmp_path / "big.jsonl", rows)
    found = detect(tmp_path)
    assert len(found.sample_rows) <= SAMPLE_ROW_LIMIT
    assert all(len(str(value)) <= SAMPLE_CELL_CHARS + 1 for row in found.sample_rows for value in row.values())


# --- rules 0-1: data Orinth itself wrote --------------------------------------
#
# These matter more than they look. Without them, re-uploading a dataset Orinth
# exported gets re-guessed from structure, and a classification dataset stored
# with `<split>/images` and `<split>/labels` is indistinguishable from a YOLO
# detection root. That is a real dataset in this repo, not a hypothetical:
# `storage/datasets/trash-clasification-*` detected as `object_detection` with
# zero labels until these rules existed.


def test_a_manifest_is_trusted_over_structure(tmp_path: Path):
    write(
        tmp_path / "manifest.json",
        json.dumps(
            {
                "task_type": "classification",
                "format": "image_folder",
                "labels": ["cardboard", "glass", "metal"],
            }
        ),
    )
    # Structure that would otherwise read as a YOLO detection root.
    image(tmp_path / "train" / "images" / "a.jpg")
    write(tmp_path / "train" / "labels" / "a.txt", "0 0.5 0.5 0.2 0.2\n")

    found = detect(tmp_path)
    assert found.task_type == "classification"
    assert found.confidence == 1.0
    assert found.candidate_labels == ["cardboard", "glass", "metal"]


def test_a_manifest_with_an_unknown_task_falls_through(tmp_path: Path):
    """A manifest from a build with a task this one dropped must not win."""
    write(tmp_path / "manifest.json", json.dumps({"task_type": "holography"}))
    for label in ("a", "b"):
        image(tmp_path / label / "0.jpg")
    found = detect(tmp_path)
    assert found.task_type == "classification"


def test_a_corrupt_manifest_falls_through_instead_of_crashing(tmp_path: Path):
    write(tmp_path / "manifest.json", "{ not json")
    for label in ("a", "b"):
        image(tmp_path / label / "0.jpg")
    assert detect(tmp_path).task_type == "classification"


def annotation(path: Path, rows: list[dict]) -> None:
    write(path, json.dumps({"annotations": rows}))


def test_text_annotation_sidecars_detect_text_classification(tmp_path: Path):
    write(tmp_path / "train" / "texts" / "a.txt", "a note")
    annotation(
        tmp_path / "train" / "annotations" / "a.json",
        [{"kind": "classification", "class_id": 0, "class_name": "Respiratory"}],
    )
    found = detect(tmp_path)
    assert found.task_type == "text_classification"
    assert found.format == "text_folder"
    assert found.candidate_labels == ["Respiratory"]
    assert found.confidence >= 0.9


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        # Both spellings occur: the schema Literal uses the short form, the
        # reference datasets on disk carry the long task-type form.
        ("summary", "summarization"),
        ("summarization", "summarization"),
        ("qa", "question_answering"),
        ("question_answering", "question_answering"),
    ],
)
def test_sidecar_kind_accepts_both_spellings(tmp_path: Path, kind: str, expected: str):
    write(tmp_path / "train" / "texts" / "a.txt", "body")
    annotation(tmp_path / "train" / "annotations" / "a.json", [{"kind": kind}])
    assert detect(tmp_path).task_type == expected


def test_image_sidecars_beat_the_yolo_layout_guess(tmp_path: Path):
    """The trash-classification case: images + labels/, but really classification."""
    image(tmp_path / "train" / "images" / "a.jpg")
    write(tmp_path / "train" / "labels" / "a.txt", "0 0.5 0.5 0.2 0.2\n")
    annotation(
        tmp_path / "train" / "annotations" / "a.json",
        [{"kind": "classification", "class_id": 0, "class_name": "glass"}],
    )
    found = detect(tmp_path)
    assert found.task_type == "classification"
    assert found.candidate_labels == ["glass"]


def test_box_and_polygon_sidecars_map_to_detection_and_segmentation(tmp_path: Path):
    image(tmp_path / "train" / "images" / "a.jpg")
    annotation(
        tmp_path / "train" / "annotations" / "a.json",
        [{"kind": "box", "class_name": "lesion"}],
    )
    assert detect(tmp_path).task_type == "object_detection"

    annotation(
        tmp_path / "train" / "annotations" / "a.json",
        [{"kind": "polygon", "class_name": "lesion"}],
    )
    assert detect(tmp_path).task_type == "segmentation"


def test_empty_sidecars_fall_through_to_structure(tmp_path: Path):
    image(tmp_path / "train" / "images" / "a.jpg")
    annotation(tmp_path / "train" / "annotations" / "a.json", [])
    found = detect(tmp_path)
    # No kind to read, so the YOLO layout rule takes it.
    assert found.format == "yolo"


# --- wrapper folders ----------------------------------------------------------


def test_a_single_wrapper_folder_is_unwrapped(tmp_path: Path):
    """People upload `my-dataset/` containing the actual dataset."""
    inner = tmp_path / "my-dataset"
    for label in ("normal", "kista"):
        image(inner / label / "0.jpg")

    found = detect(tmp_path)
    assert found.task_type == "classification"
    assert found.candidate_labels == ["kista", "normal"]
    assert any("my-dataset" in signal for signal in found.signals)


def test_the_macos_archive_folder_is_not_mistaken_for_content(tmp_path: Path):
    inner = tmp_path / "dataset"
    for label in ("a", "b"):
        image(inner / label / "0.jpg")
    (tmp_path / "__MACOSX").mkdir()

    found = detect(tmp_path)
    assert found.candidate_labels == ["a", "b"]


def test_descent_stops_at_real_content(tmp_path: Path):
    """Two sibling class folders are content, not a wrapper — do not descend."""
    for label in ("normal", "kista"):
        image(tmp_path / label / "0.jpg")
    found = detect(tmp_path)
    assert found.candidate_labels == ["kista", "normal"]
    assert not any("inside the upload" in signal for signal in found.signals)


def test_descent_stops_when_files_sit_beside_the_folder(tmp_path: Path):
    write(tmp_path / "notes.txt", "readme")
    inner = tmp_path / "images"
    image(inner / "a.jpg")
    # A file at the top level means this level is content; do not unwrap past it.
    found = detect(tmp_path)
    assert not any("inside the upload" in signal for signal in found.signals)


# --- real-world column names --------------------------------------------------
#
# Every case below is taken from a dataset actually published on Kaggle or the
# Hugging Face Hub. Tidy `text`/`label` headers are the exception in the wild,
# and matching only those produced a *half* mapping — which is worse than none,
# because apply then silently skipped every row it could not read.


def test_a_label_column_with_a_prefix_is_matched(tmp_path: Path):
    """Kaggle's airline dataset calls its label `airline_sentiment`."""
    write(
        tmp_path / "t.csv",
        "tweet_id,airline_sentiment,text\n"
        + "\n".join(f"{i},{'positive' if i % 2 else 'negative'},a tweet body {i}" for i in range(12)),
    )
    found = detect(tmp_path)
    assert found.task_type == "text_classification"
    assert set(found.candidate_labels) == {"negative", "positive"}


def test_an_unnamed_text_column_is_found_by_length(tmp_path: Path):
    """Kaggle's COVID set calls its text `OriginalTweet`; no vocabulary covers that."""
    rows = "\n".join(
        f"user{i},loc{i},this is a considerably longer piece of tweet text number {i},{'Positive' if i % 2 else 'Negative'}"
        for i in range(12)
    )
    write(tmp_path / "t.csv", f"UserName,Location,OriginalTweet,Sentiment\n{rows}")
    found = detect(tmp_path)
    assert found.task_type == "text_classification"
    assert set(found.candidate_labels) == {"Negative", "Positive"}


def test_opaque_column_names_are_resolved_by_shape(tmp_path: Path):
    """Kaggle's SMS spam set ships `v1` and `v2` and says nothing else."""
    rows = "\n".join(
        f"{'spam' if i % 3 == 0 else 'ham'},this is the full text of message number {i} which is fairly long"
        for i in range(15)
    )
    write(tmp_path / "spam.csv", f"v1,v2\n{rows}")
    found = detect(tmp_path)
    assert found.task_type == "text_classification"
    assert set(found.candidate_labels) == {"ham", "spam"}
    # Named vs inferred is reflected in the confidence, not hidden.
    assert found.confidence < 0.8


def test_a_high_cardinality_id_column_is_not_mistaken_for_a_label(tmp_path: Path):
    rows = "\n".join(f"id-{i},some text body number {i} that runs on a while,cat{i % 3}" for i in range(20))
    write(tmp_path / "t.csv", f"row_id,body,group\n{rows}")
    found = detect(tmp_path)
    assert found.task_type == "text_classification"
    assert len(found.candidate_labels) == 3


# --- parquet ------------------------------------------------------------------


def test_parquet_rows_are_read(tmp_path: Path):
    """Most Hugging Face datasets ship Parquet and nothing else."""
    polars = pytest.importorskip("polars")
    path = tmp_path / "data.parquet"
    polars.DataFrame(
        {
            "instruction": [f"question {i}" for i in range(12)],
            "input": [""] * 12,
            "output": [f"answer {i}" for i in range(12)],
        }
    ).write_parquet(path)

    found = detect(tmp_path)
    assert found.task_type == "llm_finetune"
    assert found.format == "instruction_jsonl"
    assert found.sample_rows


def test_nested_parquet_values_are_flattened_for_sniffing(tmp_path: Path):
    """SQuAD stores answers as {"text": [...], "answer_start": [...]}."""
    polars = pytest.importorskip("polars")
    path = tmp_path / "squad.parquet"
    polars.DataFrame(
        {
            "question": [f"q{i}" for i in range(10)],
            "context": [f"a context paragraph number {i}" for i in range(10)],
            "answers": [{"text": [f"a{i}"], "answer_start": [0]} for i in range(10)],
        }
    ).write_parquet(path)

    found = detect(tmp_path)
    assert found.task_type == "question_answering"
    assert any(isinstance(row.get("answers"), str) for row in found.sample_rows)


def test_an_unreadable_parquet_does_not_crash_detection(tmp_path: Path):
    (tmp_path / "broken.parquet").write_bytes(b"not actually parquet")
    found = detect(tmp_path)
    assert found.modality == "table"
    assert found.needs_input


# --- preference pairs ---------------------------------------------------------


def test_chosen_rejected_pairs_detect_as_chat_finetuning(tmp_path: Path):
    """hh-rlhf and its derivatives; the accepted side is the training target."""
    rows = "\n".join(
        json.dumps(
            {
                "chosen": f"\n\nHuman: question {i}\n\nAssistant: a good answer {i}",
                "rejected": f"\n\nHuman: question {i}\n\nAssistant: a bad answer {i}",
            }
        )
        for i in range(12)
    )
    write(tmp_path / "test.jsonl", rows)
    found = detect(tmp_path)
    assert found.task_type == "llm_finetune"
    assert found.format == "chat_jsonl"
    assert any("chosen" in signal for signal in found.signals)


# --- a JSONL that is a table, not a record shape -------------------------------


def test_a_jsonl_of_text_and_label_reads_as_text_classification(tmp_path: Path):
    """The commonest classification shape on the Hub.

    `_detect_record_files` used to recognize only the record shapes (alpaca,
    messages, ShareGPT, QA, chosen/rejected) and report anything else as "no
    recognizable structure" — so this file failed while the identical data as a
    `.csv` succeeded, because only `_detect_tables` reached `_classify_rows`.
    """
    rows = [
        {"text": f"the service was {mood} and the food arrived {speed}", "label": mood}
        for mood, speed in [("good", "fast")] * 30 + [("bad", "late")] * 30
    ]
    write(
        tmp_path / "reviews.jsonl",
        "\n".join(json.dumps(row) for row in rows) + "\n",
    )

    found = detect(tmp_path)
    assert found.task_type == "text_classification"
    assert sorted(found.candidate_labels) == ["bad", "good"]
    assert found.column_roles.get("text") == "text"
    assert found.column_roles.get("label") == "label"


def test_a_stray_jsonl_does_not_outvote_an_image_tree(tmp_path: Path):
    """Record detection runs before the image rule, so it must not claim a
    sidecar sitting beside pictures — one file would decide that a folder of
    ten thousand images is a table of rows."""
    for index in range(4):
        image(tmp_path / "cat" / f"{index}.jpg")
        image(tmp_path / "dog" / f"{index}.jpg")
    write(tmp_path / "notes.jsonl", json.dumps({"text": "a note", "label": "x"}) + "\n")

    found = detect(tmp_path)
    assert found.modality == "image"
    assert found.task_type == "classification"
    assert sorted(found.candidate_labels) == ["cat", "dog"]


def test_a_stray_jsonl_does_not_outvote_a_text_class_tree(tmp_path: Path):
    """Same guard, the other content type: `_detect_text_folders` also runs
    after the record rule."""
    for index in range(4):
        write(tmp_path / "spam" / f"{index}.txt", "buy now, limited offer")
        write(tmp_path / "ham" / f"{index}.txt", "see you at the meeting tomorrow")
    write(tmp_path / "notes.jsonl", json.dumps({"text": "a note", "label": "x"}) + "\n")

    found = detect(tmp_path)
    assert found.task_type == "text_classification"
    assert sorted(found.candidate_labels) == ["ham", "spam"]


def test_a_lone_jsonl_beside_a_readme_still_reads_as_rows(tmp_path: Path):
    """The guard is a majority, not an absolute: a dataset is not disqualified
    by shipping one text file next to its data."""
    rows = [
        {"review": f"the room was {mood} and the staff were {mood}", "verdict": mood}
        for mood in ["great"] * 30 + ["awful"] * 30
    ]
    write(tmp_path / "hotel.jsonl", "\n".join(json.dumps(row) for row in rows) + "\n")
    write(tmp_path / "README.txt", "Hotel reviews, scraped 2024.")

    found = detect(tmp_path)
    assert found.task_type == "text_classification"
