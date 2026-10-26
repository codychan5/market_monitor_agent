import pytest

from london_office_monitor.paths import INTERNAL_DIR
from london_office_monitor.seed.load import SeedError, read_internal

FRONT_MATTER = '---\ntitle: "Long note"\nkind: Note\ndate: 2025-01-06\n---\n'


def test_only_an_internal_document_over_the_documents_table_limit_is_refused(tmp_path):
    # The documents table holds at most 50,000 characters of an internal document. A longer one is named when the
    # folder is read, instead of failing the database insert halfway through a load.
    (tmp_path / "at-limit.md").write_text(FRONT_MATTER + "x" * 50_000, encoding="utf-8")
    (tmp_path / "over-limit.md").write_text(FRONT_MATTER + "x" * 50_001, encoding="utf-8")
    with pytest.raises(SeedError) as refused:
        read_internal(tmp_path)
    assert "over-limit.md" in str(refused.value) and "at-limit.md" not in str(refused.value)


def test_bundled_internal_documents_are_all_marked_mock():
    # They are written for the proof of concept and must never pass for real Nan Fung records.
    docs = read_internal(INTERNAL_DIR)
    assert len(docs) == 6
    assert all(d.title.startswith("[MOCK]") and d.synthetic and d.source_kind == "internal" for d in docs)
    assert all("MOCK INTERNAL DOCUMENT" in d.summary for d in docs)
    assert all(d.text_format == "markdown" and d.subtitle for d in docs)
