from datetime import UTC, datetime

from london_office_monitor.chunking import approx_tokens, chunk_text, contextual_header, document_key_text
from london_office_monitor.ingest.feeds import DocumentIn


def test_news_summary_is_one_chunk():
    chunks = chunk_text("A short summary. Second sentence.", fmt="plain", max_tokens=350)
    assert [(c.chunk_index, c.heading_path) for c in chunks] == [(0, ())]


def test_markdown_splits_at_headings_and_keeps_paragraphs_whole():
    text = "## What changed\nPara one.\n\nPara two.\n\n## Coming up\nPara three."
    chunks = chunk_text(text, fmt="markdown", max_tokens=350)
    assert [c.heading_path for c in chunks] == [("What changed",), ("Coming up",)]
    assert all(text[c.char_start:c.char_end].strip().endswith(".") for c in chunks)
    assert "Para one." in chunks[0].text and "Para two." in chunks[0].text


def test_nested_headings_build_a_path():
    text = "# Brief\n## Rates\nBank Rate held.\n### Gilts\nYields rose."
    assert [c.heading_path for c in chunk_text(text, fmt="markdown", max_tokens=350)] == [
        ("Brief", "Rates"), ("Brief", "Rates", "Gilts")]


def test_long_paragraph_splits_only_at_sentence_ends():
    para = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(para, fmt="plain", max_tokens=100)
    assert len(chunks) > 1 and all(c.text.endswith(".") and approx_tokens(c.text) <= 100 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chinese_sentences_split_at_full_width_stops():
    para = "".join(f"第{i}句寫字樓需求穩定。" for i in range(120))
    chunks = chunk_text(para, fmt="plain", max_tokens=50)
    assert len(chunks) > 1 and all(c.text.endswith("。") for c in chunks)


def test_offsets_reproduce_each_chunk():
    text = "First para.\n\nSecond para is here.\n\nThird."
    for c in chunk_text(text, fmt="plain", max_tokens=5):
        assert text[c.char_start:c.char_end] == c.text


def test_table_is_its_own_chunk():
    text = "Intro.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nOutro."
    assert [c.is_table for c in chunk_text(text, fmt="markdown", max_tokens=350)] == [False, True, False]


def test_contextual_header_drops_empty_parts():
    assert contextual_header("City AM", "Title", None, (), datetime(2026, 10, 21, tzinfo=UTC)) == \
        "City AM | Title | 21 Oct 2026"


def test_contextual_header_uses_the_london_date_and_heading_path():
    late_utc = datetime(2026, 10, 20, 23, 30, tzinfo=UTC)  # already 21 October in London (BST)
    assert contextual_header("Brief", "Week", "Sub", ("Rates", "Gilts"), late_utc) == \
        "Brief | Week | Sub | Rates > Gilts | 21 Oct 2026"


def test_document_key_text_joins_title_subtitle_and_summary():
    doc = DocumentIn(source_kind="news", publisher="p", title="T", summary="S", url="u",
                     published_at=datetime(2026, 10, 21, tzinfo=UTC), subtitle="Sub", meta={})
    assert document_key_text(doc) == "T\nSub\nS"
