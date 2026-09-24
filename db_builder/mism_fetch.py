"""
Fetch models from the MISM platform (https://mism-dev.renci.org) and normalise
each one into the SAME schema.org "ComputationalTool" record shape used by the
ToolDB catalog and by biomodels_fetch.py.

Every record is tagged  source: "MISM_models"  so the rest of the pipeline can
treat MISM, ToolDB and BioModels uniformly (one index, many sources).

Output: a JSON array of records, ready to be concatenated with the other
sources via merge_sources.py and fed to tooldb_prepare_docs.py / build_kg_sqlite.py.

Usage:
    python db_builder/mism_fetch.py --output data/mism_models.json
    python db_builder/mism_fetch.py --output data/mism_models.json --limit 50

Network: talks to the public MISM REST API over HTTPS. The `list models`
endpoint declares Bearer-auth in its OpenAPI schema but is reachable
unauthenticated in practice; pass --token if that ever changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

MISM_BASE = "https://mism-dev.renci.org/api/v1"
PAGE_SIZE = 100


def fetch_models(base_url: str, limit: int, token: str | None) -> list[dict]:
    """Fetch up to `limit` models, paging through the list endpoint."""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    models: list[dict] = []
    offset = 0
    total = None
    while len(models) < limit and (total is None or offset < total):
        page = min(PAGE_SIZE, limit - len(models))
        resp = requests.get(
            f"{base_url}/models",
            params={"limit": page, "offset": offset},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        total = body.get("total", 0)
        results = body.get("results", []) or []
        if not results:
            break
        models.extend(results)
        offset += len(results)
    return models[:limit]


def _authors_to_keywords(model: dict) -> list[str]:
    keywords = []
    for author in model.get("authors", []) or []:
        name = author.get("name") if isinstance(author, dict) else author
        if name:
            keywords.append(str(name))
    if model.get("organization"):
        keywords.append(str(model["organization"]))
    if model.get("license"):
        keywords.append(str(model["license"]))
    if model.get("registration_status"):
        keywords.append(str(model["registration_status"]))
    return keywords


def _publications_to_citations(model: dict) -> list[dict]:
    citations = []
    for pub in model.get("publications", []) or []:
        if not isinstance(pub, dict):
            continue
        if not (pub.get("doi") or pub.get("pmid") or pub.get("title")):
            continue
        citations.append(
            {
                "doi": pub.get("doi", ""),
                "pmid": pub.get("pmid", ""),
                "name": pub.get("title", ""),
                "abstract": "",
            }
        )
    return citations


def to_tool_record(model: dict) -> dict:
    """Map a MISM model into the schema.org ComputationalTool shape.

    Only fields the downstream pipeline reads are populated; everything is
    tagged source=MISM_models and given a stable mism_<id> tool id.
    """
    model_id = str(model.get("id") or model.get("name") or "")
    name = model.get("name") or model_id
    description = model.get("description") or f"MISM model {name}."

    # execution_type (pip/native/docker/...) plays the "how do you run this"
    # role that programmingLanguage plays for regular tools -> USES_LANGUAGE edges.
    execution_type = model.get("execution_type")

    topics = [{"name": str(scale)} for scale in (model.get("model_scales") or []) if scale]
    topics += [{"name": str(org)} for org in (model.get("organisms") or []) if org]

    external_ids = model.get("external_ids") or {}
    url = (
        external_ids.get("url")
        if isinstance(external_ids, dict) and external_ids.get("url")
        else f"https://mism-dev.renci.org/api/v1/models/{model_id}"
    )

    return {
        "@type": "ComputationalTool",
        "_id": f"mism_{model_id}",
        "identifier": model_id,
        "name": name,
        "description": description,
        "programmingLanguage": [execution_type] if execution_type else [],
        "applicationCategory": ["Multiscale simulation model"],
        "operatingSystem": [],
        "topicCategory": topics,
        "featureList": [{"name": str(tag)} for tag in (model.get("format_tags") or []) if tag],
        "keywords": _authors_to_keywords(model),
        "url": url,
        "citation": _publications_to_citations(model),
        "version": model.get("version"),
        "license": model.get("license"),
        "source": "MISM_models",  # <-- the source tag
        # The complete, untouched metadata MISM returned for this model —
        # nothing is dropped, even fields the mapping above doesn't use
        # (entry_points, containers, funding, execution_ref, etc.).
        "raw_metadata": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch MISM models → ToolDB-compatible JSON.")
    ap.add_argument("--output", default="data/mism_models.json")
    ap.add_argument("--base-url", default=MISM_BASE, help="MISM API base URL")
    ap.add_argument("--limit", type=int, default=1000, help="Max models to fetch")
    ap.add_argument("--token", default=None, help="Bearer token, if the API starts requiring one")
    args = ap.parse_args()

    print(f"[mism] fetching models from {args.base_url} (limit {args.limit})…")
    try:
        models = fetch_models(args.base_url, args.limit, args.token)
    except Exception as exc:  # noqa: BLE001
        print(f"[mism] fetch failed: {exc}", file=sys.stderr)
        raise

    records = [to_tool_record(m) for m in models]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mism] wrote {len(records)} records → {out}")


if __name__ == "__main__":
    main()
