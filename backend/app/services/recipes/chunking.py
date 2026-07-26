"""Text chunking with paragraph-aware boundaries and provenance.

Splits each source's extracted text into overlapping chunks, preferring
paragraph boundaries over hard character cuts, and records provenance
(source id, character offset) so the review table and per-chunk regeneration
know where each record came from.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    source_id: str
    offset: int
    text: str


def chunk_sources(
    sources: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Chunk ``(source_id, text)`` pairs into a flat, globally-indexed list."""
    chunk_overlap = max(0, min(chunk_overlap, max(chunk_size - 1, 0)))
    chunks: list[Chunk] = []
    for source_id, text in sources:
        for offset, piece in _chunk_text(text, chunk_size, chunk_overlap):
            chunks.append(Chunk(index=len(chunks), source_id=source_id, offset=offset, text=piece))
    return chunks


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, str]]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [(0, text)]

    step = max(1, chunk_size - chunk_overlap)
    result: list[tuple[int, str]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        # Prefer a paragraph, then sentence, then whitespace boundary near the
        # end so chunks don't slice mid-word or mid-sentence when avoidable.
        if end < length:
            end = _best_boundary(text, start, end)
        piece = text[start:end].strip()
        if piece:
            result.append((start, piece))
        if end >= length:
            break
        start = max(end - chunk_overlap, start + step)
    return result


def _best_boundary(text: str, start: int, end: int) -> int:
    # Search backwards from the hard cut for a natural break, but never accept a
    # boundary that would make the chunk trivially short.
    floor = start + max(1, (end - start) // 2)
    for marker in ("\n\n", ". ", ".\n", "\n", " "):
        idx = text.rfind(marker, floor, end)
        if idx != -1:
            return idx + len(marker)
    return end
