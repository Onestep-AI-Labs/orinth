"""Browsing and importing from the HuggingFace Hub.

Everything here is offline. The Hub itself is exercised by hand (see the phase-10
and phase-21 spec tables); what these pin down is the part that is ours — the
filter vocabulary, the projection of a Hub tag list into card metadata, and the
cell coercion without which the Hub's most-downloaded QA dataset imports zero
rows and reports success.
"""

from datetime import UTC, datetime

import pytest

from app.services import hub_facets
from app.services.dataset_hub import DatasetHubService, _as_text


class FakeCardData:
    def __init__(self, pretty_name=None, viewer=True):
        self.pretty_name = pretty_name
        self.viewer = viewer


class FakeInfo:
    """The shape `HfApi.list_datasets(full=True)` yields, reduced to what we read."""

    def __init__(self, **overrides):
        self.id = "acme/widgets"
        self.author = "acme"
        self.downloads = 1234
        self.likes = 56
        self.gated = False
        self.tags = []
        self.last_modified = datetime(2026, 1, 2, tzinfo=UTC)
        self.card_data = FakeCardData()
        self.trending_score = 7
        self.description = ""
        for key, value in overrides.items():
            setattr(self, key, value)


def project(**overrides):
    return DatasetHubService._result_from_info(FakeInfo(**overrides))


# --- the vocabulary -----------------------------------------------------------


def test_every_facet_term_carries_an_explanation():
    """The whole point of serving the vocabulary is the sentence beside each
    term. A term with an empty hint is a chip that reads as a quiz."""
    facets = hub_facets.facets()
    groups = [facets.modalities, facets.formats, facets.sizes, facets.tasks, facets.sorts]
    assert all(groups), "no facet group may be empty"
    for group in groups:
        for option in group:
            assert option.label, f"{option.value} has no label"
            assert len(option.hint) > 20, f"{option.value} needs a real explanation"


def test_every_sort_option_maps_to_a_hub_sort_key():
    """A sort the UI offers and the Hub does not understand silently falls back
    to trending, which looks like the control is broken."""
    offered = {option.value for option in hub_facets.facets().sorts}
    assert offered == set(hub_facets.SORT_KEYS)


def test_non_importable_terms_are_listed_rather_than_hidden():
    """A user looking for an image dataset should find out here that Orinth
    imports text records, not after picking one."""
    formats = {option.value: option for option in hub_facets.facets().formats}
    assert formats["imagefolder"].importable is False
    assert formats["parquet"].importable is True


@pytest.mark.parametrize(
    ("modalities", "expected"),
    [
        (["text"], True),
        (["tabular", "text"], True),
        (["image"], False),
        (["audio", "video"], False),
        # Plenty of text datasets declare no modality at all; refusing those
        # would hide most of the Hub.
        ([], True),
    ],
)
def test_importability_follows_modality(modalities, expected):
    assert hub_facets.is_importable(modalities) is expected


def test_split_tags_separates_structured_tags_from_author_keywords():
    grouped = hub_facets.split_tags(
        ["modality:text", "format:parquet", "cad", "screen-recording", "license:mit"]
    )
    assert grouped["modality"] == ["text"]
    assert grouped["format"] == ["parquet"]
    assert grouped["license"] == ["mit"]
    # Free-form keywords are real but unfilterable, so they live under "".
    assert grouped[""] == ["cad", "screen-recording"]


# --- projecting a Hub result --------------------------------------------------


def test_structured_tags_become_card_fields():
    result = project(
        tags=[
            "task_categories:question-answering",
            "language:en",
            "license:cc-by-sa-4.0",
            "size_categories:10K<n<100K",
            "format:parquet",
            "modality:text",
            "region:us",
        ]
    )
    assert result.modalities == ["text"]
    assert result.formats == ["parquet"]
    assert result.task_categories == ["question-answering"]
    assert result.languages == ["en"]
    assert result.license == "cc-by-sa-4.0"
    assert result.size_category == "10K<n<100K"
    assert result.importable is True


def test_a_card_that_disables_the_viewer_says_so():
    """`has_viewer` is the difference between a dataset you can inspect before
    importing and one you must take on trust."""
    assert project(card_data=FakeCardData(viewer=False)).has_viewer is False
    assert project(card_data=FakeCardData(viewer=True)).has_viewer is True
    # Most cards omit the key entirely, and absent means the viewer works.
    assert project(card_data=None).has_viewer is True


def test_a_video_dataset_is_browsable_but_flagged_unimportable():
    result = project(tags=["modality:video"])
    assert result.modalities == ["video"]
    assert result.importable is False


def test_the_summary_flattens_a_markdown_card_to_one_line():
    """Card descriptions open with the title repeated inside nested heading
    whitespace, so a raw prefix is mostly blanks."""
    result = project(description="\n\n\t\tDataset Card\n\n   Summary   of\n\n the thing.")
    assert result.summary == "Dataset Card Summary of the thing."


def test_a_long_summary_is_truncated_with_an_ellipsis():
    result = project(description="word " * 200)
    assert result.summary is not None
    assert len(result.summary) <= 221
    assert result.summary.endswith("…")


def test_an_empty_description_yields_no_summary_rather_than_an_empty_string():
    assert project(description="").summary is None
    assert project(description=None).summary is None


# --- feature types ------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ({"dtype": "string", "_type": "Value"}, "string"),
        ({"dtype": "int32", "_type": "Value"}, "int32"),
        ({"feature": {"dtype": "string"}, "_type": "List"}, "list"),
        ([{"dtype": "string"}], "list"),
        ({"_type": "ClassLabel", "names": ["a", "b"]}, "classlabel"),
        # SQuAD's `answers`: a bare dict of nested feature specs with no `_type`.
        # Calling it "unknown" hides exactly the case the user needs to see.
        (
            {"text": {"feature": {"dtype": "string"}}, "answer_start": {"feature": {"dtype": "int32"}}},
            "struct",
        ),
        (None, "unknown"),
    ],
)
def test_feature_types_reduce_to_one_word(spec, expected):
    """The viewer shows this beside a column name, so anything longer than a
    word competes with the name itself."""
    assert DatasetHubService._feature_type(spec) == expected


# --- cell coercion ------------------------------------------------------------


def test_a_squad_answer_struct_flattens_to_its_text():
    """The regression this exists for.

    `rajpurkar/squad` stores its answer as a struct of parallel arrays. Handing
    that dict to `_validate_record` raised for every row, so the import
    "succeeded" with 5,000 skipped and nothing imported.
    """
    assert _as_text({"text": ["Denver Broncos"], "answer_start": [177]}) == "Denver Broncos"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain", "plain"),
        (["first", "second"], "first"),
        ({"value": "v"}, "v"),
        ({"content": "c"}, "c"),
        # No recognised key: fall through to the first string-valued member.
        ({"weird": 3, "other": "found"}, "found"),
        # Nothing string-like anywhere is an unmappable row, and saying so lets
        # the importer skip it honestly.
        ({"a": 1, "b": 2}, None),
        ([], None),
    ],
)
def test_nested_cells_flatten_predictably(value, expected):
    assert _as_text(value) == expected


def test_none_survives_as_none_rather_than_the_string():
    """An absent value is a skipped row, which is correct. `"None"` would be a
    corrupt one that trains."""
    assert _as_text(None) is None


def test_a_scalar_that_is_not_text_is_left_for_validation_to_reject():
    """Coercing an int to "42" here would paper over a genuinely wrong column
    choice, and the user would find out at training time instead of import."""
    assert _as_text(42) == 42
