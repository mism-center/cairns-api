"""
Merge any number of tool sources into one JSON array for the index build.
Built-in sources today:
  1) ToolDB       (schema.org ComputationalTool records; existing catalog)
  2) BioModels    (produced by biomodels_fetch.py)
  3) MISM_models  (produced by mism_fetch.py)

Each record is guaranteed a `source` tag so the retrieval layer and evidence
cards can show which database a tool came from. Adding a new source later
needs no changes here — just pass another --source NAME=PATH.

Usage:
    python db_builder/merge_sources.py \
        --tooldb data/tooldb.json \
        --biomodels data/biomodels.json \
        --source MISM_models=data/mism_models.json \
        --output data/tools_combined.json \
        --tooldb-limit 300      # optional subset of ToolDB for fast first runs

    # Fully generic form (no built-in flags needed):
    python db_builder/merge_sources.py \
        --source tooldb=data/tooldb.json \
        --source biomodels=data/biomodels.json \
        --source MISM_models=data/mism_models.json \
        --output data/tools_combined.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str | None) -> list[dict]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"[merge] (skip, not found) {path}")
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):  # {"@graph": [...]} style
        data = data.get("@graph", [data])
    return [r for r in data if isinstance(r, dict)]


def _tag(records: list[dict], source: str) -> list[dict]:
    for r in records:
        r.setdefault("source", source)
        r["source"] = source  # a source file always wins over any pre-existing tag
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge N tool sources into one array.")
    ap.add_argument("--tooldb", default=None, help="Shortcut for --source tooldb=PATH")
    ap.add_argument("--biomodels", default=None, help="Shortcut for --source biomodels=PATH")
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeatable. Add a source as NAME=PATH, e.g. --source MISM_models=data/mism_models.json",
    )
    ap.add_argument("--output", default="data/tools_combined.json")
    ap.add_argument("--tooldb-limit", type=int, default=0, help="0 = all ToolDB records")
    args = ap.parse_args()

    sources: list[tuple[str, str]] = []
    if args.tooldb:
        sources.append(("tooldb", args.tooldb))
    if args.biomodels:
        sources.append(("biomodels", args.biomodels))
    for spec in args.source:
        if "=" not in spec:
            raise SystemExit(f"--source must be NAME=PATH, got: {spec!r}")
        name, path = spec.split("=", 1)
        sources.append((name, path))

    if not sources:
        # Back-compat default when no flags given at all.
        sources = [("tooldb", "data/tooldb.json"), ("biomodels", "data/biomodels.json")]

    merged: list[dict] = []
    counts: dict[str, int] = {}
    for name, path in sources:
        records = _tag(_load(path), name)
        if name == "tooldb" and args.tooldb_limit and args.tooldb_limit > 0:
            records = records[: args.tooldb_limit]
        counts[name] = len(records)
        merged.extend(records)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")

    counts_str = "  ".join(f"{name}={n}" for name, n in counts.items())
    print(f"[merge] {counts_str}  total={len(merged)}")
    print(f"[merge] wrote → {out}")


if __name__ == "__main__":
    main()
