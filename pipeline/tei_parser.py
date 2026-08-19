from dataclasses import dataclass

from lxml import etree

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class ParsedHeader:
    title: str | None
    authors: list[str]


@dataclass
class Reference:
    ref_id: str
    title: str | None
    authors: list[str]
    year: str | None


def _author_names(author_elems) -> list[str]:
    names = []
    for author in author_elems:
        pers = author.find("tei:persName", TEI_NS)
        if pers is None:
            continue
        forenames = [f.text for f in pers.findall("tei:forename", TEI_NS) if f.text]
        surname = pers.find("tei:surname", TEI_NS)
        surname_text = surname.text if surname is not None and surname.text else ""
        name = " ".join([*forenames, surname_text]).strip()
        if name:
            names.append(name)
    return names


def _title_with_fallback(bibl) -> str | None:
    title_elem = bibl.find(".//tei:title", TEI_NS)
    if title_elem is not None and title_elem.text:
        return title_elem.text
    # GROBID-lite occasionally misfiles the title into this note instead
    # (observed on real references — see CLAUDE.md's "Extraction quality
    # findings" entry). Falling back here recovers most of those cases.
    note_elem = bibl.find('.//tei:note[@type="report_type"]', TEI_NS)
    if note_elem is not None and note_elem.text:
        return note_elem.text
    return None


def parse_header(xml: str) -> ParsedHeader:
    root = etree.fromstring(xml.encode("utf-8"))
    title_elem = root.find(".//tei:titleStmt/tei:title", TEI_NS)
    title = title_elem.text if title_elem is not None and title_elem.text else None
    author_elems = root.findall(".//tei:sourceDesc//tei:author", TEI_NS)
    return ParsedHeader(title=title, authors=_author_names(author_elems))


def parse_references(xml: str) -> list[Reference]:
    root = etree.fromstring(xml.encode("utf-8"))
    references = []
    for bibl in root.findall(".//tei:biblStruct", TEI_NS):
        ref_id = bibl.get("{http://www.w3.org/XML/1998/namespace}id", "")
        author_elems = bibl.findall(".//tei:author", TEI_NS)
        date_elem = bibl.find(".//tei:date", TEI_NS)
        year = date_elem.get("when") if date_elem is not None else None
        references.append(Reference(
            ref_id=ref_id,
            title=_title_with_fallback(bibl),
            authors=_author_names(author_elems),
            year=year,
        ))
    return references
