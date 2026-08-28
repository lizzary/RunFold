from __future__ import annotations

import bisect
import re

from runfold_server.knowledge.models import DocumentSection, DocumentTextMatch

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_CHINESE_HEADING = re.compile(
    r"^第[零\u3007一二三四五六七八九十百千万两0-9]+"
    r"(?P<unit>部分|编|章|节|条|款)(?:\s+|[\uff1a:、.]*)"
)
_CHINESE_NUMBERED_HEADING = re.compile(
    r"^(?P<number>[一二三四五六七八九十百千万]+)[、.]\s*(?P<title>\S.*)$"
)
_DECIMAL_HEADING = re.compile(
    r"^(?P<number>\d+(?:\.\d+){0,4})(?:[、.)]|\s+)\s*(?P<title>\S.*)$"
)
_MAX_HEADING_CHARACTERS = 200


def document_sections(
    text: str, *, fallback_title: str
) -> tuple[DocumentSection, ...]:
    headings: list[tuple[int, str, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        value = line.rstrip("\r\n").strip()
        heading = _heading(value)
        if heading is not None:
            title, level = heading
            headings.append((offset, title, level))
        offset += len(line)

    if not headings:
        return (
            DocumentSection(
                section_id="section-0000",
                title=fallback_title,
                level=1,
                start_character=0,
                end_character=len(text),
            ),
        )

    starts = headings
    if headings[0][0] > 0:
        starts = [(0, "Preamble", 0), *headings]
    return tuple(
        DocumentSection(
            section_id=f"section-{index:04d}",
            title=title,
            level=level,
            start_character=start,
            end_character=starts[index + 1][0] if index + 1 < len(starts) else len(text),
        )
        for index, (start, title, level) in enumerate(starts)
    )


def slice_text(text: str, *, offset: int, maximum: int) -> tuple[str, int, bool]:
    if offset < 0 or offset > len(text):
        raise ValueError("Text offset is outside the document")
    if maximum <= 0:
        raise ValueError("Text slice size must be positive")
    end = min(len(text), offset + maximum)
    return text[offset:end], end, end == len(text)


def search_literal(
    text: str,
    *,
    query: str,
    case_sensitive: bool,
    maximum_matches: int,
    context_characters: int,
) -> tuple[int, tuple[DocumentTextMatch, ...]]:
    if not query:
        raise ValueError("Search query must not be empty")
    if maximum_matches <= 0 or context_characters < 0:
        raise ValueError("Search result limits are invalid")

    flags = 0 if case_sensitive else re.IGNORECASE
    matches = tuple(re.finditer(re.escape(query), text, flags))
    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer("\n", text))
    selected = tuple(
        DocumentTextMatch(
            start_character=match.start(),
            end_character=match.end(),
            line_number=bisect.bisect_right(line_starts, match.start()),
            context=text[
                max(0, match.start() - context_characters) : min(
                    len(text), match.end() + context_characters
                )
            ],
        )
        for match in matches[:maximum_matches]
    )
    return len(matches), selected


def _heading(value: str) -> tuple[str, int] | None:
    if not value or len(value) > _MAX_HEADING_CHARACTERS:
        return None
    markdown = _MARKDOWN_HEADING.fullmatch(value)
    if markdown is not None:
        return markdown.group(2).strip(), len(markdown.group(1))
    chinese = _CHINESE_HEADING.match(value)
    if chinese is not None:
        levels = {"编": 1, "部分": 1, "章": 1, "节": 2, "条": 3, "款": 4}
        return value, levels[chinese.group("unit")]
    chinese_numbered = _CHINESE_NUMBERED_HEADING.fullmatch(value)
    if chinese_numbered is not None:
        return value, 2
    decimal = _DECIMAL_HEADING.fullmatch(value)
    if decimal is not None:
        return value, decimal.group("number").count(".") + 1
    return None
