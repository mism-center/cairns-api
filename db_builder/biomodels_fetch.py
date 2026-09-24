"""
Fetch models from the EBI BioModels repository (https://www.biomodels.org /
https://www.ebi.ac.uk/biomodels) and normalise each one into the SAME
schema.org "ComputationalTool" record shape used by the existing ToolDB catalog.

Every record is tagged  source: "biomodels"  so the rest of the pipeline can
treat BioModels and ToolDB uniformly (one index, two sources).

Output: a JSON array of records, ready to be concatenated with the ToolDB JSON
and fed to tooldb_prepare_docs.py / build_kg_sqlite.py.

Usage:
    python db_builder/biomodels_fetch.py --output data/biomodels.json --limit 200
    python db_builder/biomodels_fetch.py --output data/biomodels.json --query "cancer" --limit 100

Network: talks to the public BioModels REST API over HTTPS (follows redirects).
No API key required.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

BIOMODELS_BASE = "https://www.ebi.ac.uk/biomodels"
HEADERS = {"Accept": "application/json", "User-Agent": "CAIRNS-API/1.0"}


def _clean_html(text: str) -> str:
    """BioModels descriptions embed SBML/XHTML notes — strip tags to plain text."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def search_model_ids(query: str, limit: int) -> list[str]:
    """Return up to `limit` BioModels IDs matching a search query."""
    ids: list[str] = []
    offset = 0
    page = min(limit, 100)
    while len(ids) < limit:
        url = (
            f"{BIOMODELS_BASE}/search?query={quote(query)}"
            f"&numResults={page}&offset={offset}&format=json"
        )
        resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", []) if isinstance(data, dict) else []
        if not models:
            break
        for m in models:
            mid = m.get("id") or m.get("submissionId")
            if mid:
                ids.append(str(mid))
        offset += page
        if len(models) < page:
            break
    return ids[:limit]


def fetch_model(model_id: str) -> dict | None:
    """Fetch one model's metadata JSON."""
    url = f"{BIOMODELS_BASE}/{quote(model_id)}?format=json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! skip {model_id}: {exc}", file=sys.stderr)
        return None


def to_tool_record(model_id: str, model: dict) -> dict:
    """Map a BioModels model into the schema.org ComputationalTool shape.

    Only fields the downstream pipeline reads are populated; everything is
    tagged source=biomodels and given a stable biomodels_<id> tool id.
    """
    name = model.get("name") or model_id
    description = _clean_html(model.get("description", "")) or f"BioModels systems-biology model {model_id}."

    fmt = model.get("format", {})
    model_format = fmt.get("name") if isinstance(fmt, dict) else (fmt or "SBML")

    # Curation status / model type become keyword-like topics.
    keywords = []
    if model.get("curationStatus"):
        keywords.append(str(model["curationStatus"]))
    if model_format:
        keywords.append(str(model_format))
    for term in model.get("modellingApproach", []) or []:
        label = term.get("name") if isinstance(term, dict) else term
        if label:
            keywords.append(str(label))

    # topicCategory from any linked ontology terms (systems biology / disease).
    topics = []
    for term in (model.get("modelTags", []) or []) + (model.get("subjects", []) or []):
        label = term.get("name") if isinstance(term, dict) else term
        if label:
            topics.append({"name": str(label)})

    urls = [f"https://www.ebi.ac.uk/biomodels/{model_id}"]

    return {
        "@type": "ComputationalTool",
        "_id": f"biomodels_{model_id.lower()}",
        "identifier": model_id,
        "name": name,
        "description": description,
        # BioModels are model files, not runnable software; record the format as
        # the "language" so KG edges (USES_LANGUAGE) still carry useful signal.
        "programmingLanguage": [model_format] if model_format else [],
        "applicationCategory": ["Systems biology model"],
        "operatingSystem": [],
        "topicCategory": topics,
        "featureList": [],
        "keywords": keywords,
        "url": urls[0],
        "source": "biomodels",  # <-- the two-source tag
        # The complete, untouched metadata BioModels returned for this model —
        # nothing is dropped, even fields the mapping above doesn't use.
        "raw_metadata": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch BioModels → ToolDB-compatible JSON.")
    ap.add_argument("--output", default="data/biomodels.json")
    ap.add_argument("--query", default="*", help="BioModels search query (default: all)")
    ap.add_argument("--limit", type=int, default=200, help="Max models to fetch")
    ap.add_argument("--sleep", type=float, default=0.1, help="Delay between model fetches (be polite)")
    args = ap.parse_args()

    print(f"[biomodels] searching '{args.query}' (limit {args.limit})…")
    ids = search_model_ids(args.query, args.limit)
    print(f"[biomodels] {len(ids)} model ids found; fetching metadata…")

    records = []
    for i, mid in enumerate(ids, 1):
        model = fetch_model(mid)
        if model is None:
            continue
        records.append(to_tool_record(mid, model))
        if i % 25 == 0:
            print(f"[biomodels] fetched {i}/{len(ids)}")
        time.sleep(args.sleep)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[biomodels] wrote {len(records)} records → {out}")


if __name__ == "__main__":
    main()
