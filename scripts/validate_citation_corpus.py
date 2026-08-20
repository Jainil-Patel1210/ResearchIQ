"""M6 corpus-wide citation validation — checks the [N] -> GROBID's
(N-1)th-reference assumption against every paper in our corpus, not just
a couple of manually spot-checked examples.

Writes:
    data/validation/citation_resolution/corpus_results.json  (source of truth)
    data/validation/citation_resolution/report.md             (generated FROM the json)

Does not run chunking/embedding — only Docling (for full-text marker
coverage) + GROBID (for references), so this is much cheaper per paper
than a full index_pdf() call.

Usage:
    python scripts/validate_citation_corpus.py
"""

import csv
import datetime
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from pipeline import citation_resolver, docling_parser, grobid_client, tei_parser

MANIFEST_PATH = Path("data/manifest.csv")
RESULTS_DIR = Path("data/validation/citation_resolution")
RESOLVER_VERSION = "m6-v1"


def _summary_from_validation(paper_id: str, v: dict) -> dict:
    checks = {
        "has_references": v["reference_count"] > 0,
        "has_markers": v["unique_markers_found"] > 0,
        "markers_within_range": not v["unresolved_markers"],
        "numbered_style_detected": v["citation_style"] == "numbered",
        "no_suspicious_brackets": v["suspicious_non_citation_brackets"] == 0,
    }
    return {
        "paper_id": paper_id,
        "status": "passed" if v["valid"] else "failed",
        "citation_style": v["citation_style"],
        "reference_count": v["reference_count"],
        "markers_found": v["unique_markers_found"],
        "unresolved_markers": v["unresolved_markers"],
        "suspicious_non_citation_brackets": v["suspicious_non_citation_brackets"],
        "checks": checks,
    }


def result_from_existing(paper_id: str) -> dict:
    """Reuse an already-computed data/references/{paper_id}.json instead
    of re-running Docling+GROBID — makes this script resumable after a
    partial/killed run."""
    with open(citation_resolver.REFERENCES_DIR / f"{paper_id}.json", encoding="utf-8") as f:
        payload = json.load(f)
    return _summary_from_validation(paper_id, payload["validation"])


def validate_one_paper(paper_id: str, pdf_path: str) -> dict:
    full_text = docling_parser.parse_pdf(pdf_path)
    refs_xml = grobid_client.fetch_references_xml(pdf_path)
    references = tei_parser.parse_references(refs_xml)
    validation = citation_resolver.validate_numbered_citations(full_text, references)

    # Persist this paper's actual reference data too, same as index_pdf()
    # would — this script gives us that as a side effect for the whole
    # corpus without needing to run the (expensive) chunk+embed steps.
    ref_map = citation_resolver.build_reference_map(references)
    payload = {
        "paper_id": paper_id,
        "citation_style": validation.citation_style,
        "validation": asdict(validation),
        "references": {str(n): asdict(ref) for n, ref in ref_map.items()},
    }
    citation_resolver.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    with open(citation_resolver.REFERENCES_DIR / f"{paper_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return _summary_from_validation(paper_id, asdict(validation))


def generate_report_md(results: dict) -> str:
    s = results["summary"]
    lines = [
        "# M6 Citation Resolution — Corpus Validation",
        "",
        f"Run: {results['run']['timestamp']} — resolver `{results['run']['resolver_version']}`",
        "",
        "## Summary",
        "",
        f"- Papers tested: {results['run']['papers_tested']}",
        f"- Passed: {s['passed']}",
        f"- Failed: {s['failed']}",
        f"- Numbered citation style: {s['numbered_style']}",
        f"- Unsupported/unknown style: {s['unsupported_style']}",
        "",
        "## Failed papers",
        "",
    ]
    failed = [p for p in results["papers"] if p["status"] == "failed"]
    if not failed:
        lines.append("None.")
    else:
        lines.append("| Paper | Style | Refs | Markers | Unresolved | Suspicious brackets |")
        lines.append("|---|---|---:|---:|---|---:|")
        for p in failed:
            lines.append(
                f"| {p['paper_id']} | {p['citation_style']} | {p['reference_count']} | "
                f"{p['markers_found']} | {p['unresolved_markers']} | "
                f"{p['suspicious_non_citation_brackets']} |"
            )
    lines.append("")
    lines.append(f"Citation resolution is valid for {s['passed']}/{results['run']['papers_tested']} papers.")
    return "\n".join(lines)


def main():
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        papers = list(csv.DictReader(f))

    print(f"validating citation resolution across {len(papers)} papers...\n", flush=True)
    per_paper = []
    for i, row in enumerate(papers, start=1):
        paper_id, pdf_path = row["paper_id"], row["pdf_path"]
        existing = citation_resolver.REFERENCES_DIR / f"{paper_id}.json"
        if existing.exists():
            print(f"[{i}/{len(papers)}] {paper_id}... reusing existing result", flush=True)
            per_paper.append(result_from_existing(paper_id))
            continue
        print(f"[{i}/{len(papers)}] {paper_id}...", end=" ", flush=True)
        try:
            result = validate_one_paper(paper_id, pdf_path)
            print(result["status"], flush=True)
        except Exception as exc:
            result = {"paper_id": paper_id, "status": "error", "error": str(exc)}
            print(f"ERROR: {exc}", flush=True)
        per_paper.append(result)

    passed = sum(1 for p in per_paper if p["status"] == "passed")
    failed = sum(1 for p in per_paper if p["status"] == "failed")
    errored = sum(1 for p in per_paper if p["status"] == "error")
    numbered = sum(1 for p in per_paper if p.get("citation_style") == "numbered")
    unsupported = sum(1 for p in per_paper if p.get("citation_style") == "unknown")

    results = {
        "run": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "papers_tested": len(papers),
            "resolver_version": RESOLVER_VERSION,
        },
        "summary": {
            "passed": passed,
            "failed": failed,
            "errored": errored,
            "numbered_style": numbered,
            "unsupported_style": unsupported,
        },
        "papers": per_paper,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "corpus_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(RESULTS_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write(generate_report_md(results))

    print(f"\n{passed}/{len(papers)} passed, {failed} failed, {errored} errored")
    print(f"results written to {RESULTS_DIR}/corpus_results.json and report.md")


if __name__ == "__main__":
    main()
