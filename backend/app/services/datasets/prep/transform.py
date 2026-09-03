"""When no rule fits the data, write code that does.

`detect.py` names a task by matching structure against rules: a `data.yaml`
beside `train/labels/`, sibling class folders, a `question`/`answer` column pair.
That covers the shapes the ML world has standardized on, and it stops dead at
the shape most real data actually arrives in — a **feature table**. Kaggle's
early-stage diabetes set is sixteen clinical yes/no columns and a `class`
column; a student-performance export is study hours, attendance, and a grade.
No column vocabulary will ever map those onto a trainable task, because the
mapping is not in the names. It is in what you decide to predict from what.

So this stage stops pattern-matching and generates a program. The column profile
goes to an OpenRouter model, which writes `transform(rows) -> rows`; the script
runs in `sandbox.py`; its output is validated against the task it claimed; and
the result is written to `_derived/` as JSONL, which apply then ingests exactly
as if the user had uploaded it that way.

Three things make that safe enough to run unattended:

**The deterministic transform is the base case, not the fallback.** `builtin`
emits the same kind of script from column statistics alone — target column,
feature columns, quartile edges baked in as a literal config — and runs first
whenever no key is configured. A model failure degrades to a duller transform,
never to none, which is the same contract `plan.py` holds for planning.

**Both engines produce a script, and both scripts run in the same sandbox.**
The builtin could have been a Python function called in-process, and that would
have been faster and would have grown a second implementation that drifts from
the one users can read. One path, one artifact: whatever the Prepare tab shows
is literally what ran.

**Output is validated against the claimed task before anything is written.** A
transform that says `text_classification` and returns rows with one distinct
label, or drops three quarters of the input, is rejected in favor of the
builtin — a model that invents a taxonomy is the damaging failure here, exactly
as in `plan.py`, and the answer is the same: check it against the data.

The tabular studio (phase 20) is still the right home for genuine tabular
modelling — feature types, imputation, a gradient-boosted trainer. This is not
that. This is the difference between a spreadsheet the platform refuses and a
spreadsheet it can train a text classifier on today.
"""

import json
import pprint
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.schemas import (
    DatasetFieldMapping,
    DatasetPrepPlan,
    DatasetPreprocessConfig,
    DatasetSplitConfig,
    PrepDecision,
    PrepTransform,
)
from app.services.datasets.constants import PREPROCESS_PRESETS
from app.services.datasets.prep.detect import (
    RECORD_SUFFIXES,
    SAMPLE_CELL_CHARS,
    TABLE_SUFFIXES,
    ColumnStats,
    Detection,
    _looks_like_classes,
    _read_rows,
    infer_label_column,
    profile_columns,
)
from app.services.datasets.prep.plan import blocking_reason
from app.services.datasets.prep.sandbox import ALLOWED_IMPORTS, run_transform
from app.services.datasets.prep.staging import (
    derived_root,
    staged_files,
    staging_root,
)

#: Suffixes this stage can read rows out of.
ROW_SUFFIXES = TABLE_SUFFIXES | RECORD_SUFFIXES

#: Rows read from one source. Matches `apply.MAX_INGEST_ROWS`, which is the real
#: ceiling — transforming rows that apply would then discard wastes sandbox time
#: and puts a misleading row count in front of the user.
MAX_TRANSFORM_ROWS = 20_000
#: Rows put in the prompt. The model needs the shape, not the data.
PROMPT_ROW_LIMIT = 12
#: More distinct labels than this and the target column is an identifier or a
#: free-text field, not a taxonomy.
MAX_LABEL_VALUES = 100
#: Below this share of input rows surviving, the transform is broken rather than
#: selective — a mapping that silently drops most of a dataset is the failure
#: mode that produced "imported 0 of 6,000 records" before.
MIN_RETENTION = 0.5
#: A numeric feature with more distinct values than this is bucketed into
#: quartiles rather than emitted verbatim: `age_41` as a bag-of-words token is
#: one useless feature per patient, while `age_q3` is a signal.
MAX_LITERAL_NUMERIC_VALUES = 12
#: How much of the gap between guessing the commonest class and being right
#: every time a single feature closes on its own. Above this, that one column
#: carries nearly all the signal in the table.
#:
#: Lift rather than raw accuracy, because raw accuracy is dominated by class
#: balance: on a 79%-Pass table every feature scores 0.79 by always answering
#: `Pass`, and the standout scores 0.96 — two numbers that look similar and mean
#: nothing alike. As lift they are 0.00 and 0.80.
#:
#: Measured at the quartile resolution the transform actually emits, which is
#: not an approximation to be improved on. Finer buckets were tried: at ten they
#: rate a genuinely derived column 0.96 and iris petal width 0.93, and at twenty
#: they rate iris petal *length* a flat 1.00. Resolution does not separate
#: leakage from a strong feature — it just shrinks every group until all of them
#: are pure. Nothing computable from the table alone makes that distinction,
#: which is why what follows reports the fact and leaves the reading to the user.
DOMINANT_FEATURE_LIFT = 0.75

#: Columns whose every value is distinct are row identifiers. They cannot help a
#: classifier and they poison a bag-of-words vocabulary, so they are dropped.
_ID_NAMES = {"id", "index", "idx", "no", "num", "number", "row", "rowid", "uuid", "key"}
_ID_SUFFIXES = ("_id", "_no", "_num", "_index", "_key", "_uid")

#: Words that mean "this column is the thing to predict". Matched per word, so
#: `exam_score` and `Pass_Fail` hit and `study_hours_per_week` does not. Plurals
#: are listed rather than stemmed: stemming makes `previous_scores` match too,
#: and the rightmost-match rule is what separates those two, not the vocabulary.
_OUTCOME_WORDS = frozenset(
    {
        "target", "outcome", "result", "results", "label", "class", "classes",
        "y", "score", "scores", "grade", "grades", "gpa", "marks", "mark",
        "performance", "rating", "risk", "status", "passed", "pass", "fail",
        "success", "final", "diagnosis", "churn", "survived", "verdict",
    }
)

#: Columns that describe *who* a row is rather than what happened to them.
#: Never wrong as features; almost always wrong as an automatically chosen
#: target, and wrong in a way that still produces a plausible-looking dataset —
#: so they are tried only after everything else has failed.
_ATTRIBUTE_WORDS = frozenset(
    {
        "gender", "sex", "race", "ethnicity", "nationality", "religion", "name",
        "country", "city", "state", "region", "date", "year", "month", "day",
        "time", "timestamp", "age",
    }
)

#: Tasks a generated transform may claim, and the row keys each one requires.
#: Nothing is accepted that does not carry them — the shape of the output is how
#: a claim gets checked.
TASK_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "text_classification": ("text", "label"),
    "summarization": ("text", "summary"),
    "question_answering": ("question", "answer"),
    "llm_finetune": ("instruction", "output"),
}
_TASK_FORMATS = {
    "text_classification": "text_folder",
    "summarization": "text_folder",
    "question_answering": "text_folder",
    "llm_finetune": "instruction_jsonl",
}


@dataclass
class TableSource:
    """One row-bearing file, read and profiled."""

    path: Path
    rows: list[dict]
    columns: list[str]
    stats: dict[str, ColumnStats]
    truncated_from: int | None = None


# --- entry point --------------------------------------------------------------


def refine_with_transform(
    plan: DatasetPrepPlan,
    detection: Detection,
    *,
    root: Path,
    api_key: str | None = None,
    model: str | None = None,
    allowed_task_types: list[str] | None = None,
    client: Any = None,
) -> DatasetPrepPlan:
    """Give a stuck plan a generated transform, or return it unchanged.

    Never raises. Called after `analyze()`, so by here the deterministic and
    model-assisted planners have both had their turn — the only two outcomes are
    a plan that can now be applied, or the same plan with a clearer reason why
    it cannot.

    "Stuck" is not only "no task was named". A plan that names a task it cannot
    feed is stuck in the way that matters more, because it applies: it writes
    items with no annotations and reports success. So the gate is whether the
    plan can actually produce examples, and a named-but-unusable task is
    replaced by a generated transform rather than left to apply.
    """
    blocker = blocking_reason(plan, detection)
    if plan.task_type and blocker is None:
        return plan

    sources = read_sources(root)
    if not sources:
        return plan

    allowed = _allowed_tasks(allowed_task_types)
    if not allowed:
        return plan

    rows = [row for source in sources for row in source.rows]
    outcome = _run_engines(
        sources,
        rows,
        allowed=allowed,
        api_key=api_key,
        model=model,
        client=client,
    )
    if outcome is None:
        return plan

    transform, produced, task, labels = outcome
    try:
        written = write_derived(root, produced)
    except OSError as error:
        plan.warnings.append(f"The prepared rows could not be written ({error}).")
        return plan

    transform.output_rows = written
    return _patched(
        plan, detection, transform, task=task, labels=labels, sources=sources,
        replaced=blocker,
    )


def _run_engines(
    sources: list[TableSource],
    rows: list[dict],
    *,
    allowed: set[str],
    api_key: str | None,
    model: str | None,
    client: Any,
) -> tuple[PrepTransform, list[dict], str, list[str]] | None:
    """Try the model's script, then the deterministic one. Report why, either way."""
    notices: list[str] = []

    if (api_key and model) or client is not None:
        attempt = _llm_attempt(
            sources, rows, allowed=allowed, api_key=api_key, model=model, client=client
        )
        if isinstance(attempt, tuple):
            return attempt
        notices.append(attempt)
    else:
        notices.append(
            "No OpenRouter key is configured, so the transform below was written from "
            "column statistics rather than by a model. Add a key in Settings for a "
            "transform tailored to this data."
        )

    builtin = _builtin_attempt(sources, rows)
    if builtin is None:
        return None
    transform, produced, task, labels = builtin
    transform.notice = " ".join(notices) or None
    return transform, produced, task, labels


# --- reading and profiling ----------------------------------------------------


def read_sources(root: Path) -> list[TableSource]:
    """Every staged row-bearing file, read in full and profiled.

    Full, not sampled: `detect` reads 500 rows because it only needs to tell a
    class column from prose, while this stage is producing the dataset itself.
    Reading the detection sample here is the bug that once ingested 20 rows of a
    52,000-row Parquet.
    """
    staging = staging_root(root)
    sources: list[TableSource] = []
    for path in staged_files(root):
        if path.suffix.lower() not in ROW_SUFFIXES:
            continue
        rows = _read_rows(path, limit=None)
        if not rows:
            continue
        truncated = None
        if len(rows) > MAX_TRANSFORM_ROWS:
            truncated = len(rows)
            rows = rows[:MAX_TRANSFORM_ROWS]
        columns = _union_columns(rows)
        if not columns:
            continue
        sources.append(
            TableSource(
                path=path.relative_to(staging) if staging in path.parents else path,
                rows=rows,
                columns=columns,
                stats=profile_columns(rows, columns),
                truncated_from=truncated,
            )
        )
    return sources


def _union_columns(rows: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(str(key), None)
    return list(seen)


# --- the deterministic engine -------------------------------------------------


def _builtin_attempt(
    sources: list[TableSource], rows: list[dict]
) -> tuple[PrepTransform, list[dict], str, list[str]] | None:
    columns = _union_columns(rows)
    stats = profile_columns(rows[: max(500, len(rows) // 10)] or rows, columns)
    choice = choose_target(columns, stats, rows)
    if choice is None:
        return None
    target, buckets, rationale = choice

    features = [
        column
        for column in columns
        if column != target and not _is_identifier(column, stats[column], len(rows))
    ]
    if not features:
        return None

    bucketed = _bucketing(features, rows, stats)
    code = builtin_code(target=target, features=features, bucketed=bucketed, label_buckets=buckets)
    notes = dominant_feature_notes(target, features, bucketed, rows)

    result = run_transform(code, rows)
    if not result.ok:
        return None
    produced, labels, _ = _validate("text_classification", result.rows, len(rows))
    if produced is None or labels is None:
        return None

    return (
        PrepTransform(
            engine="builtin",
            code=code,
            source_files=[str(source.path) for source in sources],
            input_rows=len(rows),
            target_column=target,
            feature_columns=features,
            rationale=" ".join([rationale, *notes]),
        ),
        produced,
        "text_classification",
        labels,
    )


def choose_target(
    columns: list[str], stats: dict[str, ColumnStats], rows: list[dict]
) -> tuple[str, list[float] | None, str] | None:
    """Which column is the thing to predict.

    Not "the one with the fewest classes", which was the first thing tried and
    is wrong in a way that looks right: on a student-performance export it picks
    `gender` over `result`, produces a perfectly valid dataset, and trains a
    model for a question nobody asked. Tables carry two much better signals, and
    both are conventions rather than statistics:

    1. **The name.** `class`, `result`, `outcome`, `target`, `exam_score` — a
       column named after an outcome is one. Where several match, the rightmost
       wins, because `previous_scores` precedes `exam_score` for a reason.
    2. **The position.** The target is conventionally the last column. That is
       true of essentially every UCI and Kaggle table, including both of the
       ones this stage was built against.

    Only when neither speaks does column shape decide, and there the demographic
    attributes are tried last: predicting `gender` from study habits is the
    failure this ordering exists to avoid.

    A chosen column that is continuous is bucketed into thirds. That changes the
    question from regression to three-class classification, so it is stated in
    the returned rationale rather than left for the user to infer from a
    suspiciously round accuracy.
    """
    usable = [
        column
        for column in columns
        if not _is_identifier(column, stats[column], len(rows))
    ]
    if not usable:
        return None

    for column, why in _declared_targets(usable):
        choice = _target_from(column, rows, stats, why)
        if choice is not None:
            return choice

    # Nothing declared itself. Fall back to shape, demographics last.
    for pool in (
        [column for column in usable if not _is_attribute(column)],
        usable,
    ):
        if not pool:
            continue
        categorical = infer_label_column(
            pool, {name: stats[name] for name in pool}, exclude=set()
        )
        if categorical:
            choice = _target_from(categorical, rows, stats, "shape")
            if choice is not None:
                return choice
    return None


def _declared_targets(usable: list[str]) -> list[tuple[str, str]]:
    """Columns that say they are the target, most credible first."""
    named = [
        (column, "name")
        for column in reversed(usable)
        if _OUTCOME_WORDS & _words(column)
    ]
    last = usable[-1]
    if all(column != last for column, _ in named):
        named.append((last, "position"))
    return named


def _target_from(
    column: str, rows: list[dict], stats: dict[str, ColumnStats], why: str
) -> tuple[str, list[float] | None, str] | None:
    """Turn a chosen column into a target, bucketing it if it is continuous."""
    because = {
        "name": "it is named like an outcome",
        "position": "it is the last column, where tables conventionally put the target",
        "shape": "it holds a small closed set of repeated values",
    }[why]

    if _looks_like_classes(stats[column]):
        values = sorted(
            {str(row[column]) for row in rows if row.get(column) not in (None, "")}
        )
        return (
            column,
            None,
            f"`{column}` was read as the column to predict because {because}; it holds "
            f"{len(values)} value(s) ({', '.join(values[:6])}). Every other column "
            "becomes a feature token.",
        )

    if not _is_numeric(column, rows):
        return None
    edges = _quantile_edges(_numbers(column, rows), 3)
    if not edges:
        return None
    return (
        column,
        edges,
        f"`{column}` was read as the column to predict because {because}, and it is "
        f"continuous — so it is cut at {edges[0]:.4g} and {edges[1]:.4g} into low / "
        "medium / high. That turns a regression into a three-class problem; check the "
        "thresholds before trusting the result.",
    )


def builtin_code(
    *,
    target: str,
    features: list[str],
    bucketed: dict[str, list[float]],
    label_buckets: list[float] | None,
) -> str:
    """The deterministic transform, emitted as source with its choices baked in.

    Written as text rather than called as a function so that the script in the
    Prepare tab is the script that ran. The token form is `column_value`
    (`polyuria_yes`, `gender_male`) rather than a readable `Polyuria: Yes`
    sentence for a mechanical reason: the NLP preprocessing preset strips
    punctuation with `[^\\w\\s]`, which would sever every column from its value
    and leave a bag of bare `yes`/`no` tokens with the pairing destroyed.
    Underscores survive that regex, so the binding does too.
    """
    # `pformat`, not `json.dumps`: the config is baked into Python source, and
    # JSON spells `None` as `null`, which is a `NameError` the moment the script
    # loads. The two notations agree often enough that the difference only shows
    # up on the rows a bucketed target produces.
    config = pprint.pformat(
        {
            "target": target,
            "features": features,
            "bucketed": bucketed,
            "label_buckets": label_buckets,
        },
        width=84,
        sort_dicts=False,
    )
    return _BUILTIN_TEMPLATE.replace("__CONFIG__", config)


_BUILTIN_TEMPLATE = '''"""Written by Orinth from the column statistics of this upload.

Each row becomes one text-classification example: the target column supplies the
label, and every other column becomes a `column_value` token. Numeric columns
with many distinct values are bucketed into quartiles, because one token per
distinct number carries no signal a bag-of-words classifier can use.
"""

import re

CONFIG = __CONFIG__

_BUCKET_NAMES = ["low", "medium", "high"]


def _slug(value):
    """Lowercase, underscore-joined. Survives `remove_punctuation` intact."""
    return re.sub(r"[^0-9a-z]+", "_", str(value).strip().lower()).strip("_")


def _number(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _bucket(value, edges, names=None):
    number = _number(value)
    if number is None:
        return None
    for index, edge in enumerate(edges):
        if number <= edge:
            return names[index] if names else "q%d" % (index + 1)
    return names[len(edges)] if names else "q%d" % (len(edges) + 1)


def _label_of(row):
    raw = row.get(CONFIG["target"])
    if raw is None or str(raw).strip() == "":
        return None
    if CONFIG["label_buckets"]:
        return _bucket(raw, CONFIG["label_buckets"], _BUCKET_NAMES)
    return str(raw).strip()


def transform(rows):
    prepared = []
    for row in rows:
        label = _label_of(row)
        if not label:
            continue
        tokens = []
        for column in CONFIG["features"]:
            value = row.get(column)
            if value is None or str(value).strip() == "":
                continue
            edges = CONFIG["bucketed"].get(column)
            token = _bucket(value, edges) if edges else _slug(value)
            if not token:
                continue
            tokens.append("%s_%s" % (_slug(column), token))
        if not tokens:
            continue
        prepared.append({"text": " ".join(tokens), "label": label})
    return prepared
'''


# --- the model engine ---------------------------------------------------------

SYSTEM_PROMPT = (
    "You write small, dependency-free Python data transforms for a machine "
    "learning platform. You are shown the column profile of a table a user "
    "uploaded that the platform could not map to a training task on its own. "
    "You reply with ONLY a JSON object — no prose, no markdown fences."
)


def build_prompt(sources: list[TableSource], allowed: set[str], rows: list[dict]) -> str:
    source = sources[0]
    stats = source.stats
    lines = [
        f"The upload holds {len(rows):,} rows across {len(sources)} file(s).",
        f"Columns of `{source.path}`:",
    ]
    for column in source.columns:
        stat = stats[column]
        lines.append(
            f"  - `{column}`: {stat.distinct} distinct value(s), "
            f"{stat.filled} filled, ~{stat.mean_chars:.0f} characters per value"
        )
    lines += [
        "",
        "Sample rows:",
        json.dumps(_prompt_rows(source.rows), ensure_ascii=False, default=str),
        "",
        f"Tasks this project can train: {json.dumps(sorted(allowed))}",
        "Required output keys per task: "
        + json.dumps({task: list(keys) for task, keys in TASK_REQUIREMENTS.items()}),
        "",
        "Write `transform(rows)`, taking the list of raw row dicts above and "
        "returning a list of dicts carrying exactly the keys the task you chose "
        "requires. Keep every row you can; dropping rows loses training data.",
        "",
        "The script runs in a sandbox with no filesystem, no network, and no "
        "third-party packages. Importable modules: "
        + ", ".join(sorted(ALLOWED_IMPORTS))
        + ". Anything else raises ImportError.",
        "",
        "For a feature table, the useful shape is one token per column bound to "
        "its value (`age_q3 gender_male polyuria_yes`) with the target column as "
        "the label — downstream preprocessing strips punctuation, so bind column "
        "to value with an underscore rather than a colon or a space. Drop "
        "identifier columns. Bucket continuous features rather than emitting one "
        "token per distinct number.",
        "",
        'Return {"task_type": one of the allowed tasks, "code": "<the full '
        'Python source>", "rationale": "<one sentence>", "target_column": '
        '"<column the label came from, or null>", "warnings": [string]}.',
    ]
    return "\n".join(lines)


def _prompt_rows(rows: list[dict]) -> list[dict]:
    """A prompt-safe slice: few rows, short cells.

    Cell truncation is not cosmetic. One HTML blob or one 40kB review body in
    the sample would spend the context window on a single row and leave the
    model reading fewer columns than the table has.
    """
    return [
        {
            str(key): (
                value[:SAMPLE_CELL_CHARS] + "…"
                if isinstance(value, str) and len(value) > SAMPLE_CELL_CHARS
                else value
            )
            for key, value in row.items()
        }
        for row in rows[:PROMPT_ROW_LIMIT]
    ]


def _llm_attempt(
    sources: list[TableSource],
    rows: list[dict],
    *,
    allowed: set[str],
    api_key: str | None,
    model: str | None,
    client: Any,
) -> tuple[PrepTransform, list[dict], str, list[str]] | str:
    """The model's script, or a sentence saying why the builtin ran instead."""
    from app.services.providers import (
        ChatMessage,
        LlmAuthError,
        LlmError,
        LlmRateLimitError,
        OpenRouterClient,
    )

    resolved = client or OpenRouterClient(api_key=api_key)
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=build_prompt(sources, allowed, rows)),
    ]
    try:
        body, usage = resolved.chat_json(messages, model=model or "", temperature=0.0)
    except LlmAuthError:
        return "OpenRouter rejected the configured API key, so the deterministic transform ran instead."
    except LlmRateLimitError:
        return "OpenRouter is rate limiting, so the deterministic transform ran instead."
    except LlmError as error:
        return f"The OpenRouter call failed ({error}), so the deterministic transform ran instead."

    task = body.get("task_type")
    code = body.get("code")
    if not isinstance(code, str) or not code.strip():
        return "The model returned no transform code, so the deterministic transform ran instead."
    if not isinstance(task, str) or task not in allowed:
        return (
            f"The model proposed `{task}`, which this project cannot train, so the "
            "deterministic transform ran instead."
        )

    result = run_transform(code, rows)
    if not result.ok:
        return f"The generated transform did not run ({result.error}), so the deterministic transform ran instead."

    produced, labels, reason = _validate(task, result.rows, len(rows))
    if produced is None:
        return f"The generated transform {reason}, so the deterministic transform ran instead."

    target = body.get("target_column")
    target = target if isinstance(target, str) and target in sources[0].columns else None
    features = [column for column in sources[0].columns if column != target]
    # The same check the deterministic engine runs. A model picking the target
    # does not make a column that dominates it any less worth mentioning.
    notes = (
        dominant_feature_notes(
            target, features, _bucketing(features, rows, sources[0].stats), rows
        )
        if target and task == "text_classification"
        else []
    )
    return (
        PrepTransform(
            engine="llm",
            code=code,
            source_files=[str(source.path) for source in sources],
            input_rows=len(rows),
            target_column=target,
            feature_columns=features,
            rationale=" ".join([str(body.get("rationale") or "").strip(), *notes]).strip(),
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            estimated_cost_usd=usage.cost_usd,
        ),
        produced,
        task,
        labels or [],
    )


# --- validating what came back ------------------------------------------------


def _validate(
    task: str, rows: list[dict], input_count: int
) -> tuple[list[dict] | None, list[str] | None, str]:
    """Keep only rows that carry the task's keys, then judge the whole batch.

    Row-level filtering is not enough on its own. A transform that returns a
    hundred well-formed rows out of five thousand has not been selective, it has
    been wrong, and importing its output would produce a dataset that trains and
    means nothing.
    """
    required = TASK_REQUIREMENTS.get(task)
    if not required:
        return None, None, f"claimed `{task}`, which is not a task this stage produces"

    kept = [
        {key: value for key, value in row.items() if key in required or key == "input"}
        for row in rows
        if all(str(row.get(key) or "").strip() for key in required)
    ]
    if not kept:
        return None, None, f"returned no rows carrying {' and '.join(f'`{k}`' for k in required)}"
    if input_count and len(kept) < input_count * MIN_RETENTION:
        return None, None, (
            f"kept only {len(kept):,} of {input_count:,} rows, which loses most of the dataset"
        )

    if task != "text_classification":
        return kept, [], ""

    labels = sorted({str(row["label"]).strip() for row in kept})
    if len(labels) < 2:
        return None, None, "produced a single label, which nothing can be trained to distinguish"
    if len(labels) > MAX_LABEL_VALUES:
        return None, None, (
            f"produced {len(labels):,} distinct labels, so the target column holds "
            "identifiers or free text rather than classes"
        )
    return kept, labels, ""


# --- writing and patching -----------------------------------------------------


def write_derived(root: Path, rows: list[dict]) -> int:
    """Replace `_derived/` with this run's output.

    Replaced, never appended to: a second prep run over the same upload must
    produce the dataset its own plan describes, not that one merged with the
    previous plan's leftovers.
    """
    import shutil

    derived = derived_root(root)
    shutil.rmtree(derived, ignore_errors=True)
    derived.mkdir(parents=True, exist_ok=True)
    target = derived / "prepared.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return len(rows)


def _patched(
    plan: DatasetPrepPlan,
    detection: Detection,
    transform: PrepTransform,
    *,
    task: str,
    labels: list[str],
    sources: list[TableSource],
    replaced: str | None = None,
) -> DatasetPrepPlan:
    """Fold a successful transform into the plan it unblocked."""
    if replaced:
        plan.warnings.append(
            f"The {str(plan.task_type).replace('_', ' ')} plan was replaced by a "
            f"generated transform: {replaced}."
        )
    plan.transform = transform
    plan.task_type = task  # type: ignore[assignment]
    plan.format = _TASK_FORMATS[task]  # type: ignore[assignment]
    plan.labels = labels
    plan.field_mapping = _mapping_for_task(task)
    plan.needs_input = None
    plan.confidence = max(plan.confidence, 0.6 if transform.engine == "llm" else 0.5)
    plan.preprocess = DatasetPreprocessConfig(
        enabled=True, preset="nlp_clean", transforms=list(PREPROCESS_PRESETS["nlp_clean"])
    )
    plan.split = DatasetSplitConfig(
        train=0.7, valid=0.2, test=0.1, seed=42, stratify=task == "text_classification"
    )

    engine_label = "a model" if transform.engine == "llm" else "column statistics"
    plan.rationale = (
        f"{detection.modality.capitalize()} data with no task Orinth recognizes, so a "
        f"Python transform written from {engine_label} reshaped "
        f"{transform.input_rows:,} rows into {transform.output_rows:,} "
        f"{task.replace('_', ' ')} examples. " + (transform.rationale or "")
    ).strip()

    if transform.notice:
        plan.warnings.append(transform.notice)
    for source in sources:
        if source.truncated_from:
            plan.warnings.append(
                f"`{source.path}` holds {source.truncated_from:,} rows; the first "
                f"{MAX_TRANSFORM_ROWS:,} were prepared."
            )
    plan.warnings.append(
        "The original upload is kept unchanged; Undo restores the dataset to how it "
        "arrived."
    )

    plan.decisions.append(
        PrepDecision(
            field="transform",
            value={
                "engine": transform.engine,
                "target_column": transform.target_column,
                "output_rows": transform.output_rows,
            },
            source="llm" if transform.engine == "llm" else "heuristic",
            confidence=plan.confidence,
            rationale=transform.rationale
            or "A generated Python transform reshaped the rows into training examples.",
            evidence=(
                f"{transform.input_rows:,} rows in `"
                + "`, `".join(transform.source_files[:3])
                + f"` → {transform.output_rows:,} examples."
            ),
        )
    )
    plan.decisions.append(
        PrepDecision(
            field="task_type",
            value=task,
            source="llm" if transform.engine == "llm" else "heuristic",
            confidence=plan.confidence,
            rationale=(
                f"The transform emits {', '.join(TASK_REQUIREMENTS[task])}, which is a "
                f"{task.replace('_', ' ')} example."
            ),
            evidence=f"{transform.output_rows:,} prepared rows.",
        )
    )
    if labels:
        plan.decisions.append(
            PrepDecision(
                field="labels",
                value=labels,
                source="detected",
                confidence=plan.confidence,
                rationale=(
                    f"The {len(labels)} distinct values the transform produced, taken "
                    "from the data rather than named by a model."
                ),
                evidence=f"Labels: {', '.join(labels[:8])}.",
            )
        )
    return plan


def _mapping_for_task(task: str) -> DatasetFieldMapping:
    if task == "text_classification":
        return DatasetFieldMapping(text="text", label="label")
    if task == "summarization":
        return DatasetFieldMapping(text="text", summary="summary")
    if task == "question_answering":
        return DatasetFieldMapping(question="question", answer="answer")
    return DatasetFieldMapping(instruction="instruction", input="input", output="output")


def _allowed_tasks(allowed_task_types: list[str] | None) -> set[str]:
    available = set(TASK_REQUIREMENTS)
    if allowed_task_types is None:
        return available
    return available & set(allowed_task_types)


# --- column helpers -----------------------------------------------------------


def _words(column: str) -> set[str]:
    return {word for word in re.split(r"[^a-z0-9]+", column.lower()) if word}


def _is_identifier(column: str, stat: ColumnStats, row_count: int) -> bool:
    """A column that names rows rather than describing them.

    Both tests are needed. `student_id` is caught by name even when a small
    sample happens to repeat a value, and an unnamed column whose every value is
    distinct is an identifier whatever it is called.
    """
    lowered = column.lower().strip()
    if lowered in _ID_NAMES or lowered.startswith("unnamed"):
        return True
    if any(lowered.endswith(suffix) for suffix in _ID_SUFFIXES):
        return True
    if _ID_NAMES & _words(column) and stat.distinct > max(2, row_count // 4):
        return True
    return stat.filled >= 8 and stat.distinct == stat.filled


def dominant_feature_notes(
    target: str,
    features: list[str],
    bucketed: dict[str, list[float]],
    rows: list[dict],
) -> list[str]:
    """Name any feature that on its own nearly determines the target.

    Reported as an observation, not a verdict. A column computed from the target
    and a column that is simply the best predictor in the dataset look identical
    from here — iris petal width is not leakage — and the wording says which
    question the reader has to answer. Suggesting a drop outright would, on the
    average table, delete the most useful column in it.

    Measured over the *bucketed* values the transform emits. Raw values are
    meaningless here: a continuous column puts one row in every group, so every
    group is trivially pure and every continuous feature scores 1.0.
    """
    labels = [str(row.get(target) or "").strip() for row in rows]
    total = sum(1 for label in labels if label)
    if total < 20:
        return []

    majority = max((labels.count(label) for label in set(labels) if label), default=0)
    base = majority / total
    if base >= 0.999:
        return []

    warnings: list[str] = []
    for column in features:
        groups: dict[str, dict[str, int]] = {}
        for row, label in zip(rows, labels, strict=True):
            if not label:
                continue
            key = _grouping_key(row.get(column), bucketed.get(column))
            if key is None:
                continue
            counts = groups.setdefault(key, {})
            counts[label] = counts.get(label, 0) + 1
        if len(groups) < 2:
            continue
        covered = sum(sum(counts.values()) for counts in groups.values())
        if not covered:
            continue
        accuracy = sum(max(counts.values()) for counts in groups.values()) / covered
        if (accuracy - base) / (1 - base) >= DOMINANT_FEATURE_LIFT:
            warnings.append(
                f"`{column}` alone predicts `{target}` for {accuracy:.0%} of rows, "
                f"against {base:.0%} for always guessing the commonest class — it carries "
                "nearly all the signal here. That is what you want from a real feature and "
                "what leakage looks like too, so check whether it was computed from "
                f"`{target}` before reading much into the accuracy."
            )
    return warnings[:3]


def _grouping_key(value: Any, edges: list[float] | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    if not edges:
        return str(value).strip()
    number = _as_float(value)
    if number is None:
        return None
    for index, edge in enumerate(edges):
        if number <= edge:
            return f"q{index + 1}"
    return f"q{len(edges) + 1}"


def _is_attribute(column: str) -> bool:
    return bool(_ATTRIBUTE_WORDS & _words(column))


def _is_numeric(column: str, rows: list[dict]) -> bool:
    values = [row.get(column) for row in rows]
    present = [value for value in values if value not in (None, "")]
    if not present:
        return False
    parsed = sum(1 for value in present if _as_float(value) is not None)
    return parsed >= len(present) * 0.9


def _numbers(column: str, rows: list[dict]) -> list[float]:
    numbers = [_as_float(row.get(column)) for row in rows]
    return [number for number in numbers if number is not None]


def _as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _quantile_edges(numbers: list[float], buckets: int) -> list[float] | None:
    """Cut points splitting `numbers` into equal-sized buckets.

    Distinct edges are required: a column where three quarters of the rows share
    one value produces duplicate cut points, and a bucket no row can fall into is
    a class with no examples — which is what makes a split silently untrainable.
    """
    ordered = sorted(numbers)
    if len(ordered) < buckets * 2:
        return None
    edges = [ordered[int(len(ordered) * index / buckets)] for index in range(1, buckets)]
    return edges if len(set(edges)) == len(edges) else None


def _bucketing(
    features: list[str], rows: list[dict], stats: dict[str, ColumnStats]
) -> dict[str, list[float]]:
    """Quartile cut points for every feature that needs them, keyed by column."""
    return {
        column: edges
        for column in features
        if column in stats and (edges := _bucket_edges(column, rows, stats[column]))
    }


def _bucket_edges(column: str, rows: list[dict], stat: ColumnStats) -> list[float] | None:
    if stat.distinct <= MAX_LITERAL_NUMERIC_VALUES:
        return None
    if not _is_numeric(column, rows):
        return None
    return _quantile_edges(_numbers(column, rows), 4)
