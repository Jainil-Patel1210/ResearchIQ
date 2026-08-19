import os

import requests

GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")


def _post_pdf(endpoint: str, pdf_path, **extra_fields) -> str:
    url = f"{GROBID_URL}/api/{endpoint}"
    with open(pdf_path, "rb") as f:
        response = requests.post(url, files={"input": f}, data=extra_fields, timeout=120)
    response.raise_for_status()
    return response.text


def fetch_header_xml(pdf_path) -> str:
    return _post_pdf("processHeaderDocument", pdf_path)


def fetch_references_xml(pdf_path) -> str:
    return _post_pdf("processReferences", pdf_path, consolidateCitations="0")
