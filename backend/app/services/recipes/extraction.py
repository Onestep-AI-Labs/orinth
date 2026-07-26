"""Per-source text extraction for data recipes.

Pure functions over raw file bytes, one per accepted type. Extraction runs at
upload time; each returns plain text plus per-source stats (pages, characters,
warnings). Heavy ML libraries are never imported here — only pypdf/python-docx
(small pure-Python cores) and the stdlib.

A PDF with no extractable text (scanned or encrypted) returns empty text and a
warning rather than raising: it is excluded from generation but does not fail
the recipe. OCR is out of scope (phase-11 spec).
"""

import csv
import io
import json
from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    text: str = ""
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def characters(self) -> int:
        return len(self.text)


def extract_text(filename: str, data: bytes, suffix: str) -> ExtractionResult:
    suffix = suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix == ".docx":
        return _extract_docx(data)
    if suffix == ".csv":
        return _extract_csv(data)
    if suffix == ".jsonl":
        return _extract_jsonl(data)
    # .txt / .md and any other plain-text upload.
    return _extract_plain(data)


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _extract_plain(data: bytes) -> ExtractionResult:
    return ExtractionResult(text=_decode(data).strip())


def _extract_pdf(data: bytes) -> ExtractionResult:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except Exception:  # noqa: BLE001 — dependency import guard.
        return ExtractionResult(warnings=["PDF support is unavailable on this server"])
    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, Exception) as exc:  # noqa: BLE001
        return ExtractionResult(warnings=[f"Could not read PDF: {exc}"])
    if reader.is_encrypted:
        # An empty-password decrypt succeeds for many "encrypted" PDFs; only warn
        # when it genuinely cannot be opened.
        try:
            reader.decrypt("")
        except Exception:  # noqa: BLE001
            return ExtractionResult(
                pages=len(reader.pages),
                warnings=["PDF is encrypted; no text could be extracted"],
            )
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 — one bad page must not fail the source.
            parts.append("")
    text = "\n\n".join(part.strip() for part in parts if part.strip()).strip()
    warnings: list[str] = []
    if not text:
        warnings.append("No extractable text found (scanned or image-only PDF); excluded from generation")
    return ExtractionResult(text=text, pages=len(reader.pages), warnings=warnings)


def _extract_docx(data: bytes) -> ExtractionResult:
    try:
        import docx
    except Exception:  # noqa: BLE001
        return ExtractionResult(warnings=["DOCX support is unavailable on this server"])
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        return ExtractionResult(warnings=[f"Could not read DOCX: {exc}"])
    parts = [para.text for para in document.paragraphs if para.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    text = "\n\n".join(parts).strip()
    warnings = [] if text else ["No text found in DOCX"]
    return ExtractionResult(text=text, warnings=warnings)


def _extract_csv(data: bytes) -> ExtractionResult:
    raw = _decode(data)
    try:
        reader = csv.reader(io.StringIO(raw))
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    except Exception:  # noqa: BLE001
        return ExtractionResult(text=raw.strip(), warnings=["CSV could not be parsed; used raw text"])
    if not rows:
        return ExtractionResult(warnings=["CSV is empty"])
    header, *body = rows
    lines: list[str] = []
    for row in body:
        pairs = [
            f"{header[i].strip()}: {value.strip()}"
            for i, value in enumerate(row)
            if i < len(header) and value.strip()
        ]
        if pairs:
            lines.append("; ".join(pairs))
    text = "\n".join(lines).strip() or raw.strip()
    return ExtractionResult(text=text)


def _extract_jsonl(data: bytes) -> ExtractionResult:
    raw = _decode(data)
    lines: list[str] = []
    skipped = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        lines.append(_stringify_json(obj))
    text = "\n\n".join(part for part in lines if part).strip()
    warnings = [f"{skipped} malformed JSONL lines skipped"] if skipped else []
    if not text and not warnings:
        warnings.append("JSONL is empty")
    return ExtractionResult(text=text, warnings=warnings)


def _stringify_json(obj: object) -> str:
    if isinstance(obj, dict):
        return "\n".join(f"{key}: {value}" for key, value in obj.items() if str(value).strip())
    if isinstance(obj, list):
        return "\n".join(_stringify_json(item) for item in obj)
    return str(obj).strip()
