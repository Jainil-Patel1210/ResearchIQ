"""Unit tests for pipeline/citation_resolver.py's pure functions — no
GROBID, no network, no real PDFs. Corpus-level validation against real
papers lives in scripts/validate_citation_corpus.py instead; these tests
cover edge cases deterministically and fast.
"""

from pipeline.citation_resolver import (
    build_reference_map,
    count_suspicious_brackets,
    extract_citation_markers,
    resolve,
    validate_numbered_citations,
)
from pipeline.tei_parser import Reference


def make_ref(n: int) -> Reference:
    return Reference(ref_id=f"b{n - 1}", title=f"Paper {n}", authors=[f"Author {n}"], year="2020")


def test_extract_single_marker():
    assert extract_citation_markers("as shown in [13]") == [13]


def test_extract_comma_separated_markers():
    assert extract_citation_markers("prior work [3, 7]") == [3, 7]


def test_extract_comma_no_space():
    assert extract_citation_markers("prior work [3,7]") == [3, 7]


def test_extract_many_markers_one_bracket():
    assert extract_citation_markers("[4, 27, 28, 22]") == [4, 27, 28, 22]


def test_extract_multiple_brackets():
    assert extract_citation_markers("see [1] and also [2, 3]") == [1, 2, 3]


def test_extract_no_markers():
    assert extract_citation_markers("no citations here") == []


def test_build_reference_map_is_one_indexed():
    refs = [make_ref(1), make_ref(2), make_ref(3)]
    ref_map = build_reference_map(refs)
    assert ref_map[1].ref_id == "b0"
    assert ref_map[3].ref_id == "b2"
    assert 0 not in ref_map


def test_resolve_returns_matching_references_in_order():
    ref_map = build_reference_map([make_ref(1), make_ref(2), make_ref(3)])
    resolved = resolve("cites [2] and [3]", ref_map)
    assert [r.ref_id for r in resolved] == ["b1", "b2"]


def test_resolve_skips_duplicates():
    ref_map = build_reference_map([make_ref(1)])
    resolved = resolve("[1] ... later again [1]", ref_map)
    assert len(resolved) == 1


def test_resolve_skips_out_of_range_silently():
    ref_map = build_reference_map([make_ref(1)])
    resolved = resolve("[1] and also [99]", ref_map)
    assert [r.ref_id for r in resolved] == ["b0"]


def test_count_suspicious_brackets_flags_equation():
    assert count_suspicious_brackets("as derived in Equation [12]") == 1


def test_count_suspicious_brackets_flags_figure_abbreviation():
    assert count_suspicious_brackets("shown in Fig. [3]") == 1


def test_count_suspicious_brackets_does_not_flag_real_citation():
    assert count_suspicious_brackets("prior work has shown [12] that...") == 0


def test_validate_numbered_citations_valid_paper():
    refs = [make_ref(i) for i in range(1, 21)]
    text = "background [1] and method [5, 12] and results [20]"
    result = validate_numbered_citations(text, refs)
    assert result.citation_style == "numbered"
    assert result.valid is True
    assert result.unresolved_markers == []


def test_validate_numbered_citations_out_of_range_marker():
    refs = [make_ref(i) for i in range(1, 6)]  # only 5 references
    text = "cites [1] and [2] and an impossible [99]"
    result = validate_numbered_citations(text, refs)
    assert result.valid is False
    assert result.unresolved_markers == [99]


def test_validate_numbered_citations_no_markers_is_unknown_style():
    refs = [make_ref(1), make_ref(2)]
    result = validate_numbered_citations("no bracketed citations at all", refs)
    assert result.citation_style == "unknown"
    assert result.valid is False


def test_validate_numbered_citations_missing_marker_one_is_unknown_style():
    # e.g. author-year style papers can still contain stray bracketed
    # numbers (equation refs etc) without ever citing [1] specifically.
    refs = [make_ref(i) for i in range(1, 21)]
    text = "see Equation [5] and Table [8]"
    result = validate_numbered_citations(text, refs)
    assert result.citation_style == "unknown"
