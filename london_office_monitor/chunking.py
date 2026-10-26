"""Structure-first chunking (owner decision D4, option C): split at headings, then paragraphs, merge paragraphs up to a
token budget, split an over-long paragraph only at sentence ends, and keep tables whole."""
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from .clock import LONDON

if TYPE_CHECKING:
    from .ingest.feeds import DocumentIn

# A rough, model-independent token estimate: about four characters of English per token.
CHARS_PER_TOKEN = 4
HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
TABLE_LINE = re.compile(r"^\s*\|")
# A sentence ends at . ! ? followed by whitespace or the end of the text, or at a full-width 。！？ (Chinese text
# puts no space after them).
SENTENCE_END = re.compile(r"[.!?](?=\s|$)|[。！？]")


@dataclass(frozen=True)
class ChunkIn:
    chunk_index: int
    heading_path: tuple[str, ...]
    text: str
    char_start: int
    char_end: int
    is_table: bool


@dataclass(frozen=True)
class _Block:
    heading_path: tuple[str, ...]
    start: int
    end: int
    is_table: bool


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _blocks(text: str, fmt: Literal["plain", "markdown"]) -> list[_Block]:
    """Paragraph and table blocks with their offsets in text and the heading path they sit under."""
    blocks: list[_Block] = []
    headings: list[tuple[int, str]] = []  # (level, title) from the outermost heading in
    path: tuple[str, ...] = ()
    start = end = None
    table = False

    def flush() -> None:
        nonlocal start, end
        if start is not None:
            blocks.append(_Block(path, start, end, table))
        start = end = None

    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        line_start, line_end = offset, offset + len(content)
        offset += len(line)
        heading = HEADING.match(content) if fmt == "markdown" else None
        if heading:
            flush()
            level = len(heading.group(1))
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, heading.group(2)))
            path = tuple(title for _, title in headings)
            continue
        if not content.strip():
            flush()
            continue
        is_table_line = fmt == "markdown" and bool(TABLE_LINE.match(content))
        if start is not None and is_table_line != table:
            flush()
        if start is None:
            start, table = line_start + (len(content) - len(content.lstrip())), is_table_line
        end = line_end
    flush()
    return blocks


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """(start, end) of each sentence in text[start:end], trimmed of surrounding whitespace."""
    spans, cursor = [], start
    for match in SENTENCE_END.finditer(text, start, end):
        stop = match.end()
        spans.append((cursor, stop))
        cursor = stop
    if cursor < end and text[cursor:end].strip():
        spans.append((cursor, end))
    trimmed = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        if s < e:
            trimmed.append((s, e))
    return trimmed


def chunk_text(text: str, *, fmt: Literal["plain", "markdown"], max_tokens: int) -> list[ChunkIn]:
    """Chunks whose text is exactly text[char_start:char_end]. A single sentence longer than the budget is kept
    whole: cutting it would make a passage that no longer says what the source said."""
    spans: list[tuple[tuple[str, ...], int, int, bool]] = []
    current: list | None = None  # [path, start, end]

    def close() -> None:
        nonlocal current
        if current is not None:
            spans.append((current[0], current[1], current[2], False))
            current = None

    def add(path: tuple[str, ...], s: int, e: int) -> None:
        nonlocal current
        if current is not None and (current[0] != path or approx_tokens(text[current[1]:e]) > max_tokens):
            close()
        if current is None:
            current = [path, s, e]
        else:
            current[2] = e

    for block in _blocks(text, fmt):
        if block.is_table:
            close()
            spans.append((block.heading_path, block.start, block.end, True))
        elif approx_tokens(text[block.start:block.end]) <= max_tokens:
            add(block.heading_path, block.start, block.end)
        else:
            close()
            for s, e in _sentence_spans(text, block.start, block.end):
                add(block.heading_path, s, e)
            close()
    close()
    return [ChunkIn(chunk_index=i, heading_path=path, text=text[s:e], char_start=s, char_end=e, is_table=is_table)
            for i, (path, s, e, is_table) in enumerate(spans)]


def contextual_header(publisher: str, title: str, subtitle: str | None, heading_path: tuple[str, ...],
                      published_at: datetime) -> str:
    """The line placed before a chunk's text when embedding it, so a passage carries where it came from."""
    local = published_at.astimezone(LONDON)
    parts = [publisher, title, subtitle or "", " > ".join(heading_path), f"{local.day} {local:%b %Y}"]
    return " | ".join(p for p in parts if p)


def document_key_text(doc: "DocumentIn") -> str:
    """The document key: what finds the article as a whole (title, subtitle and summary)."""
    return "\n".join(p for p in (doc.title, doc.subtitle, doc.summary) if p)
