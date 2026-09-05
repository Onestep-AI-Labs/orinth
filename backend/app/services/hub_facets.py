"""The filter vocabulary of the Hub, and what each term actually means.

The Hub tags a dataset with `modality:text`, `format:parquet`,
`size_categories:10K<n<100K`, `task_categories:question-answering`. Those strings
are precise and almost entirely opaque to someone who has not used the Hub
before — "imagefolder" and "webdataset" are not self-describing, and neither is
`n<1K`.

So the vocabulary lives here rather than being hardcoded in the browser, with a
one-line explanation beside every term. The UI renders `hint` as the tooltip on
the filter chip *and* on the badge that shows the same term on a result card, so
the explanation is identical wherever the word appears and there is exactly one
place to correct it.

`importable` is the part that is genuinely ours rather than the Hub's, and it is
now a much shorter list of exclusions than it was. The as-is ingest
(`dataset_hub.ingest_hub`) downloads a split in the shape the Hub served it and
hands it to the prep agent, so image, tabular, and text datasets all import —
an image split becomes a folder tree named by class, everything else becomes
JSONL. What stays browse-only is what the platform has no task for at all: audio,
video, 3D, and the shard formats the preview server cannot read row-wise.

Saying so on the chip is still the point: a filter that returns things the user
cannot use is worse than one that tells them why up front, and the alternative —
discovered the hard way — is an import button that fails after the click.
"""

from app.schemas import DatasetHubFacetOption, DatasetHubFacets

#: Hub `modality:` values. The ones Orinth has a task for import as-is; the
#: others are listed rather than hidden, so a user looking for an audio dataset
#: finds out here rather than after picking one.
MODALITIES = [
    ("text", "Text", "Sentences, documents, or chat turns. Imports as records or text items.", True),
    ("tabular", "Tabular", "Rows and columns — a spreadsheet. Imports as rows; Orinth works out the columns.", True),
    ("image", "Image", "Pictures. Imports as an image dataset, one folder per class.", True),
    ("audio", "Audio", "Sound files. Orinth has no audio task yet, so this is browse only.", False),
    ("video", "Video", "Video files. Orinth has no video task yet, so this is browse only.", False),
    ("document", "Document", "PDFs and scans. Build a dataset from these under Documents instead.", False),
    ("geospatial", "Geospatial", "Coordinates and map layers. Browse only.", False),
    ("3d", "3D", "Meshes and point clouds. Browse only.", False),
    ("timeseries", "Time-series", "Measurements over time. Imports as rows when a column holds text.", True),
]

#: Hub `format:` values. The distinction that matters to a user is "can the
#: preview server read this", which tracks the row-oriented formats.
FORMATS = [
    ("parquet", "Parquet", "Columnar binary format. The Hub's default — fastest to preview and import.", True),
    ("json", "JSON", "One JSON object per record, or a JSON array. Reads directly.", True),
    ("csv", "CSV", "Comma-separated rows. Reads directly.", True),
    ("text", "Text", "Plain text files, one document each.", True),
    ("arrow", "Arrow", "In-memory columnar format. Reads through the preview server.", True),
    ("imagefolder", "Image folder", "Images in class-named directories. Imports as an image dataset.", True),
    ("soundfolder", "Sound folder", "Audio in class-named directories. Browse only.", False),
    ("webdataset", "WebDataset", "Sharded tar archives for streaming. The preview server cannot read these row-wise, so browse only.", False),
]

#: Hub `size_categories:` values, verbatim — the Hub matches on the literal
#: string including the `<` characters, so these cannot be prettified server-side.
SIZES = [
    ("n<1K", "Under 1K rows", "Tiny. Fine for a smoke test, too small to fine-tune on.", True),
    ("1K<n<10K", "1K – 10K rows", "A comfortable fine-tuning set, and it imports in seconds.", True),
    ("10K<n<100K", "10K – 100K rows", "Large. Choose how many rows to take when you import.", True),
    ("100K<n<1M", "100K – 1M rows", "Very large. Import reads from the start of the split.", True),
    ("1M<n<10M", "Over 1M rows", "Huge. Preview works; import samples the head.", True),
]

#: Hub `task_categories:` values, narrowed to the ones Orinth has a task for.
#: The full Hub list runs to dozens; the rest are still shown as badges on a
#: card (`task_label` gives them a readable name) but are not offered as filters.
TASKS = [
    ("text-generation", "Text generation", "Prompt-and-continuation records. The usual shape for instruction tuning.", True),
    ("question-answering", "Question answering", "Question, context, and answer columns.", True),
    ("summarization", "Summarization", "A document and its summary.", True),
    ("text-classification", "Text classification", "Text with a label. Imports as a text classifier dataset.", True),
    ("text2text-generation", "Text-to-text", "Paired input and target text.", True),
    ("conversational", "Conversational", "Multi-turn dialogue. Maps to chat records.", True),
    ("translation", "Translation", "Source and target language pairs.", True),
    ("image-classification", "Image classification", "Pictures with one class each. Imports as an image dataset.", True),
    ("object-detection", "Object detection", "Pictures with boxes. Imports as images; boxes need annotating in Orinth.", True),
    ("image-segmentation", "Image segmentation", "Pictures with masks. Imports as images; masks need annotating in Orinth.", True),
]

#: Hub sort keys. `trendingScore` is what huggingface.co itself defaults to, so
#: an unfiltered browse here shows the same datasets the website does.
SORTS = [
    ("trending", "Trending", "What the Hub is featuring right now — the huggingface.co default.", True),
    ("downloads", "Most downloaded", "All-time download count. Biased toward old, established datasets.", True),
    ("likes", "Most liked", "Community favourites.", True),
    ("modified", "Recently updated", "Most recently changed, newest first.", True),
]

#: `sort` value → the key `HfApi.list_datasets` expects. Kept beside the
#: vocabulary so adding a sort is one edit, not two in different files.
SORT_KEYS = {
    "trending": "trendingScore",
    "downloads": "downloads",
    "likes": "likes",
    "modified": "lastModified",
}


def _options(rows: list[tuple[str, str, str, bool]]) -> list[DatasetHubFacetOption]:
    return [
        DatasetHubFacetOption(value=value, label=label, hint=hint, importable=importable)
        for value, label, hint, importable in rows
    ]


def facets() -> DatasetHubFacets:
    return DatasetHubFacets(
        modalities=_options(MODALITIES),
        formats=_options(FORMATS),
        sizes=_options(SIZES),
        tasks=_options(TASKS),
        sorts=_options(SORTS),
    )


def _label_index(rows: list[tuple[str, str, str, bool]]) -> dict[str, str]:
    return {value: label for value, label, _, _ in rows}


_MODALITY_LABELS = _label_index(MODALITIES)
_FORMAT_LABELS = _label_index(FORMATS)
_SIZE_LABELS = _label_index(SIZES)
_TASK_LABELS = _label_index(TASKS)

#: Modalities Orinth has a task for. Anything else makes a result browsable but
#: not importable, which the card says on its face.
_IMPORTABLE_MODALITIES = {"text", "tabular", "image", "timeseries"}


def split_tags(tags: list[str]) -> dict[str, list[str]]:
    """Group `prefix:value` Hub tags by prefix, dropping the free-form ones.

    A dataset carries twenty-odd tags of which maybe eight are structured. The
    rest are author keywords ("cad", "screen-recording") that mean nothing to a
    filter, so they are separated out under `""` and shown only on the detail
    screen.
    """
    grouped: dict[str, list[str]] = {}
    for tag in tags:
        prefix, separator, value = tag.partition(":")
        key = prefix if separator else ""
        if not value and separator:
            continue
        grouped.setdefault(key, []).append(value if separator else prefix)
    return grouped


def modality_label(value: str) -> str:
    return _MODALITY_LABELS.get(value, value)


def format_label(value: str) -> str:
    return _FORMAT_LABELS.get(value, value)


def size_label(value: str) -> str:
    return _SIZE_LABELS.get(value, value)


def task_label(value: str) -> str:
    #: Hub task ids are kebab-case; anything outside our narrowed list still gets
    #: a readable label rather than being dropped, because the tag is real even
    #: when it is not filterable.
    return _TASK_LABELS.get(value, value.replace("-", " "))


def is_importable(modalities: list[str]) -> bool:
    """Whether Orinth has any task that could hold this dataset.

    A dataset with no declared modality is treated as importable: plenty of text
    datasets simply omit the tag, and refusing those would hide most of the Hub.
    """
    if not modalities:
        return True
    return any(modality in _IMPORTABLE_MODALITIES for modality in modalities)
