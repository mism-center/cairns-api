from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


TEXT_KEYS = (
    "name",
    "description",
    "identifier",
    "url",
    "@id",
    "id",
    "value",
    "label",
    "title",
)

URL_KEYS = ("url", "mainEntityOfPage", "codeRepository", "softwareHelp", "downloadUrl")


def load_json_array(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("@graph"), list):
            return [item for item in data["@graph"] if isinstance(item, dict)]
        return [data]
    return []


def ensure_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _compact(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in TEXT_KEYS:
            if key in value and value.get(key) is not None:
                text = as_text(value.get(key))
                if text:
                    return text
        return ""
    return ""


def collect_text_list(value: Any) -> list[str]:
    values = []
    for item in ensure_list(value):
        if isinstance(item, dict):
            text = as_text(item)
            if text:
                values.append(text)
        elif isinstance(item, list):
            values.extend(collect_text_list(item))
        else:
            text = as_text(item)
            if text:
                values.append(text)
    return dedupe_keep_order(values)


def dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def normalize_tool_id(record: dict) -> str:
    return (
        as_text(record.get("_id"))
        or as_text(record.get("identifier"))
        or as_text(record.get("name"))
    )


def normalize_entity(item: Any) -> dict:
    if isinstance(item, str):
        return {"id": item, "url": item if item.startswith("http") else "", "name": item}
    if not isinstance(item, dict):
        text = as_text(item)
        return {"id": text, "url": "", "name": text}
    identifier = as_text(item.get("identifier")) or as_text(item.get("@id")) or as_text(item.get("id"))
    url = as_text(item.get("url")) or as_text(item.get("@id"))
    name = as_text(item.get("name")) or as_text(item.get("label")) or identifier or url
    return {"id": identifier or url or name, "url": url, "name": name}


def extract_urls(record: dict) -> list[str]:
    urls = []
    for key in URL_KEYS:
        urls.extend(collect_text_list(record.get(key)))
    return dedupe_keep_order(urls)


def extract_citation_abstracts(record: dict) -> list[str]:
    abstracts = []
    for key in ("citation", "citedBy"):
        for citation in ensure_list(record.get(key)):
            if isinstance(citation, dict):
                abstract = as_text(citation.get("abstract"))
                if abstract:
                    abstracts.append(abstract)
    return dedupe_keep_order(abstracts)


def extract_papers(record: dict) -> list[dict]:
    papers = []
    for key in ("citation", "citedBy"):
        for citation in ensure_list(record.get(key)):
            if not isinstance(citation, dict):
                continue
            doi = as_text(citation.get("doi"))
            pmid = as_text(citation.get("pmid"))
            if not doi and not pmid:
                continue
            paper_id = f"doi:{doi}" if doi else f"pmid:{pmid}"
            papers.append(
                {
                    "paper_id": paper_id,
                    "doi": doi,
                    "pmid": pmid,
                    "title": as_text(citation.get("name")) or as_text(citation.get("title")),
                    "abstract": as_text(citation.get("abstract")),
                }
            )
    deduped = {}
    for paper in papers:
        deduped[paper["paper_id"]] = paper
    return list(deduped.values())


def _extract_io_entries(record: dict, io_key: str) -> list[dict]:
    entries = []
    for io_item in ensure_list(record.get(io_key)):
        io_name = as_text(io_item.get("name")) if isinstance(io_item, dict) else as_text(io_item)
        formats = []
        if isinstance(io_item, dict):
            for fmt in ensure_list(io_item.get("encodingFormat")):
                entity = normalize_entity(fmt)
                if entity["id"] or entity["name"]:
                    formats.append(entity)
        entries.append({"name": io_name, "formats": formats, "io_type": io_key})
    return entries


def extract_io_data(record: dict) -> dict:
    inputs = _extract_io_entries(record, "input")
    outputs = _extract_io_entries(record, "output")
    io_formats = dedupe_keep_order(
        [fmt["name"] for group in (inputs, outputs) for io_item in group for fmt in io_item["formats"] if fmt["name"]]
    )
    return {
        "inputs": inputs,
        "outputs": outputs,
        "io_formats": io_formats,
    }


def chunk_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [text]
    words = text.split()
    if not words:
        return [text]

    chunks: list[str] = []
    current = ""
    for word in words:
        tentative = word if not current else f"{current} {word}"
        if len(tentative) > max_chars and current:
            chunks.append(current)
            current = word
        else:
            current = tentative
    if current:
        chunks.append(current)
    return chunks


def build_tool_documents(record: dict, chunk_size: int = 0) -> list[dict]:
    tool_id = normalize_tool_id(record)
    tool_name = as_text(record.get("name")) or tool_id
    identifier = as_text(record.get("identifier"))
    description = as_text(record.get("description"))
    application_category = collect_text_list(record.get("applicationCategory"))
    programming_language = collect_text_list(record.get("programmingLanguage"))
    operating_system = collect_text_list(record.get("operatingSystem"))
    keywords = collect_text_list(record.get("keywords"))
    topics = [normalize_entity(item) for item in ensure_list(record.get("topicCategory"))]
    operations = [normalize_entity(item) for item in ensure_list(record.get("featureList"))]
    io_data = extract_io_data(record)
    urls = extract_urls(record)
    citation_abstracts = extract_citation_abstracts(record)

    fields_used = []
    canonical_parts = []

    def add_field(field_name: str, values: list[str]):
        compact_values = dedupe_keep_order([_compact(v) for v in values if _compact(v)])
        if not compact_values:
            return
        fields_used.append(field_name)
        canonical_parts.append(f"{field_name}: " + "; ".join(compact_values))

    add_field("name", [tool_name])
    add_field("identifier", [identifier])
    add_field("description", [description])
    add_field("applicationCategory", application_category)
    add_field("programmingLanguage", programming_language)
    add_field("operatingSystem", operating_system)
    add_field("keywords", keywords)
    add_field("topicCategory", [topic["name"] for topic in topics if topic["name"]])
    add_field("featureList", [op["name"] for op in operations if op["name"]])
    add_field(
        "io",
        [
            f"{io_item['io_type']}:{io_item['name']} ({fmt['name']})"
            for io_group in (io_data["inputs"], io_data["outputs"])
            for io_item in io_group
            for fmt in io_item["formats"]
            if fmt["name"]
        ],
    )
    add_field("urls", urls)
    add_field("citation_abstracts", citation_abstracts)

    canonical_text = "\n".join(canonical_parts).strip()
    chunks = chunk_text(canonical_text, chunk_size)
    chunk_count = len(chunks)

    payload = {
        "tool_id": tool_id,
        "tool_name": tool_name,
        "identifier": identifier,
        "description": description,
        # Preserve the record's own source tag (tooldb | biomodels); default tooldb.
        "source": record.get("source", "tooldb"),
        "fields_used": fields_used,
        "urls": urls,
        "license": record.get("license"),
        "isAccessibleForFree": record.get("isAccessibleForFree"),
        "applicationCategory": application_category,
        "programmingLanguage": programming_language,
        "operatingSystem": operating_system,
        "edam_topics": dedupe_keep_order([topic["name"] for topic in topics if topic["name"]]),
        "edam_ops": dedupe_keep_order([op["name"] for op in operations if op["name"]]),
        "io_formats": io_data["io_formats"],
        # The complete source record, unfiltered — includes raw_metadata (the
        # untouched source-API response) for sources that provide it, so no
        # field is ever lost between the source and the API response.
        "metadata": record,
    }

    docs = []
    for index, chunk in enumerate(chunks):
        docs.append(
            {
                "doc_id": f"{tool_id}#{index + 1}",
                "tool_id": tool_id,
                "tool_name": tool_name,
                "chunk_index": index + 1,
                "chunk_count": chunk_count,
                "canonical_text": chunk,
                "payload": payload,
            }
        )
    return docs
