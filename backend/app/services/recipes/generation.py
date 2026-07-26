"""Record generation: deterministic rule-based scaffolding and LLM prompting.

Pure, side-effect-free helpers. The service layer (``service.py``) owns the
run loop, cancellation, status, and OpenRouter calls; this module only turns a
chunk of text into candidate record dicts and builds/parses the LLM prompt.

Rule output is structurally valid scaffolding for human review — not a quality
substitute for LLM generation (phase-11 spec).
"""

import json
import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---- shared text helpers ----------------------------------------------------


def _paragraphs(text: str) -> list[str]:
    parts = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(parts) <= 1:
        parts = [line.strip() for line in text.splitlines() if line.strip()] or ([text.strip()] if text.strip() else [])
    return parts


def _head_body(paragraph: str) -> tuple[str, str]:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return paragraph.strip(), paragraph.strip()
    head = lines[0]
    body = " ".join(lines[1:]).strip()
    return head, body or paragraph.strip()


def _is_question(line: str) -> bool:
    return line.rstrip().endswith("?")


def _is_heading(line: str) -> bool:
    stripped = line.rstrip()
    return 0 < len(stripped) <= 80 and not stripped.endswith((".", "!", "?", ",", ";", ":"))


def _first_sentences(text: str, count: int = 2) -> str:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    joined = " ".join(sentences[:count]).strip()
    return joined or text.strip()


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact[:limit].rstrip() + ("…" if len(compact) > limit else "")


def _topic(paragraph: str) -> str:
    head, _ = _head_body(paragraph)
    if _is_heading(head):
        return head
    return _shorten(paragraph, 60)


# ---- rule-based generation --------------------------------------------------


def rule_records(chunk_text: str, output_format: str, flavor: str, count: int) -> list[dict]:
    paragraphs = _paragraphs(chunk_text)
    if not paragraphs:
        return []
    records: list[dict] = []
    for paragraph in paragraphs:
        if len(records) >= count:
            break
        record = _rule_record(paragraph, output_format, flavor)
        if record is not None:
            records.append(record)
    return records


def _rule_record(paragraph: str, output_format: str, flavor: str) -> dict | None:
    head, body = _head_body(paragraph)
    if output_format == "chat_jsonl":
        user, assistant = _rule_turn(paragraph, head, body, flavor)
        if not assistant.strip():
            return None
        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
    instruction, context, output = _rule_instruction(paragraph, head, body, flavor)
    if not output.strip():
        return None
    record: dict = {"instruction": instruction, "output": output}
    if context:
        record["input"] = context
    return record


def _rule_instruction(paragraph: str, head: str, body: str, flavor: str) -> tuple[str, str, str]:
    if flavor == "instruction":
        # Frame a title/section as the task, the passage as the response.
        if _is_heading(head) and body != paragraph:
            return f"Write a detailed explanation of: {head}", "", body
        return "Expand the following note into a complete passage.", _shorten(paragraph, 120), paragraph
    # qa (and conversation-as-instruction fallback): produce a question/answer.
    if _is_question(head):
        return head, "", body
    return (
        f"According to the source, what does it say about {_topic(paragraph)}?",
        "",
        _first_sentences(paragraph, 3),
    )


def _rule_turn(paragraph: str, head: str, body: str, flavor: str) -> tuple[str, str]:
    if _is_question(head):
        return head, body
    if flavor == "instruction":
        return f"Tell me about {_topic(paragraph)}.", paragraph
    return f"Can you explain {_topic(paragraph)}?", _first_sentences(paragraph, 3)


# ---- LLM prompting ----------------------------------------------------------


_INSTRUCTION_SHAPE = (
    'Each array item must be an object: '
    '{"instruction": "<task or question>", "input": "<optional context, may be empty>", '
    '"output": "<the answer or completion>"}.'
)
_CHAT_SHAPE = (
    'Each array item must be an object: '
    '{"messages": [{"role": "user", "content": "<user turn>"}, '
    '{"role": "assistant", "content": "<assistant reply>"}]}.'
)

_FLAVOR_GUIDANCE = {
    "qa": "Write natural question-and-answer pairs a reader could ask about the passage.",
    "instruction": "Write instruction/response pairs that ask a model to produce or explain the content.",
    "conversation": "Write short, natural single-turn conversations grounded in the passage.",
}

SYSTEM_PROMPT = (
    "You generate supervised fine-tuning data from a source passage. "
    "Respond with ONLY a JSON array — no prose, no markdown fences. "
    "Ground every record strictly in the passage; do not invent facts."
)


def build_prompt(chunk_text: str, output_format: str, flavor: str, count: int) -> tuple[str, str]:
    shape = _CHAT_SHAPE if output_format == "chat_jsonl" else _INSTRUCTION_SHAPE
    guidance = _FLAVOR_GUIDANCE.get(flavor, _FLAVOR_GUIDANCE["qa"])
    user = (
        f"Produce up to {count} records as a JSON array. {shape} {guidance}\n\n"
        f"Passage:\n\"\"\"\n{chunk_text}\n\"\"\""
    )
    return SYSTEM_PROMPT, user


def build_repair_prompt(previous: str, output_format: str, count: int) -> tuple[str, str]:
    shape = _CHAT_SHAPE if output_format == "chat_jsonl" else _INSTRUCTION_SHAPE
    user = (
        "Your previous reply was not valid JSON matching the required shape. "
        f"Return ONLY a JSON array of up to {count} records. {shape}\n\n"
        f"Previous reply:\n{previous}"
    )
    return SYSTEM_PROMPT, user


def parse_llm_records(content: str, output_format: str) -> list[dict] | None:
    """Extract and shallow-validate a JSON array of records; ``None`` if malformed."""
    data = _extract_json_array(content)
    if data is None:
        return None
    records: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if output_format == "chat_jsonl":
            if isinstance(item.get("messages"), list):
                records.append({"messages": item["messages"]})
        else:
            if item.get("instruction") and item.get("output"):
                record = {"instruction": item["instruction"], "output": item["output"]}
                if item.get("input"):
                    record["input"] = item["input"]
                records.append(record)
    return records or None


def _extract_json_array(content: str) -> list | None:
    text = content.strip()
    if text.startswith("```"):
        # Strip a ```json … ``` fence if the model added one anyway.
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None
