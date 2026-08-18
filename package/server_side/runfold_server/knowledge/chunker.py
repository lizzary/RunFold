from __future__ import annotations

import hashlib
import re

from runfold_server.knowledge.models import Chunk

_HORIZONTAL_WHITESPACE = re.compile(r"[\t\f\v ]+")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


def chunk_text(text: str, *, size: int, overlap: int) -> tuple[Chunk, ...]:
    normalized = _normalized_for_chunks(text)
    if not normalized:
        return ()

    values: list[str] = []
    start = 0
    while start < len(normalized):
        hard_end = min(start + size, len(normalized))
        end = hard_end
        if hard_end < len(normalized):
            minimum_break = start + max(1, size // 2)
            paragraph_break = normalized.rfind("\n\n", minimum_break, hard_end)
            line_break = normalized.rfind("\n", minimum_break, hard_end)
            space_break = normalized.rfind(" ", minimum_break, hard_end)
            end = max(paragraph_break + 2, line_break + 1, space_break + 1)
            if end <= start:
                end = hard_end
        value = normalized[start:end].strip()
        if value:
            values.append(value)
        if end >= len(normalized):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and normalized[next_start].isspace():
            next_start += 1
        start = next_start

    return tuple(
        Chunk(
            chunk_id=hashlib.sha256(f"{ordinal}\0{value}".encode()).hexdigest(),
            ordinal=ordinal,
            text=value,
        )
        for ordinal, value in enumerate(values)
    )


def _normalized_for_chunks(text: str) -> str:
    lines = (_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    return _EXCESS_NEWLINES.sub("\n\n", "\n".join(lines)).strip()
