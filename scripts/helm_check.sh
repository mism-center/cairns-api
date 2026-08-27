#!/usr/bin/env bash
# Smoke-check the helm chart: lint + render the three configurations that matter.
# Asserts the subpath wiring (ROOT_PATH == ingress path, Prefix match, no rewrite).
set -euo pipefail
CHART="$(cd "$(dirname "${BASH_SOURCE[0]}")/../helm/cairns-api" && pwd)"

# Subchart tarballs are gitignored; fetch them from Chart.lock if missing.
[ -d "$CHART/charts" ] || helm dependency build "$CHART" >/dev/null

helm lint "$CHART" >/dev/null
echo "[ok] lint"

# 1) default: subpath /cairns
out="$(helm template t "$CHART")"
grep -q 'ROOT_PATH: "/cairns"'   <<<"$out" || { echo "FAIL: ROOT_PATH not set"; exit 1; }
grep -q 'path: /cairns'          <<<"$out" || { echo "FAIL: ingress path"; exit 1; }
grep -q 'pathType: Prefix'       <<<"$out" || { echo "FAIL: pathType"; exit 1; }
! grep -q 'rewrite-target'       <<<"$out" || { echo "FAIL: rewrite breaks root_path"; exit 1; }
echo "[ok] subpath /cairns"

# 2) root mount: root_path must be "" (not "/"), else FastAPI double-prefixes URLs
out="$(helm template t "$CHART" --set ingress.path=/ )"
grep -q 'ROOT_PATH: ""'          <<<"$out" || { echo "FAIL: root mount ROOT_PATH"; exit 1; }
echo "[ok] root mount"

# 3) ollama backend + existing secret + index job
out="$(helm template t "$CHART" \
        --set llm.backend=ollama \
        --set llm.openai.existingSecret=my-llm-secret \
        --set indexBuild.enabled=true)"
grep -q 'OLLAMA_BASE_URL'        <<<"$out" || { echo "FAIL: ollama env"; exit 1; }
grep -q 'name: my-llm-secret'    <<<"$out" || { echo "FAIL: existingSecret"; exit 1; }
grep -q 'kind: Job'              <<<"$out" || { echo "FAIL: index job"; exit 1; }
echo "[ok] ollama + secret + build job"

# 4) bundled Qdrant subchart: URL must point at the in-release Service
out="$(helm template t "$CHART" --set qdrant.enabled=true)"
grep -q 'QDRANT_URL: "http://t-qdrant:6333"' <<<"$out" || { echo "FAIL: bundled qdrant URL"; exit 1; }
grep -q 'name: t-qdrant'                     <<<"$out" || { echo "FAIL: qdrant subchart not rendered"; exit 1; }
echo "[ok] bundled qdrant"

# 5) external Qdrant (default): subchart must stay out of the release
out="$(helm template t "$CHART" --set vectorStore.url=http://qdrant.infra:6333)"
grep -q 'QDRANT_URL: "http://qdrant.infra:6333"' <<<"$out" || { echo "FAIL: external qdrant URL"; exit 1; }
! grep -q 'app.kubernetes.io/name: qdrant'       <<<"$out" || { echo "FAIL: subchart rendered while disabled"; exit 1; }
echo "[ok] external qdrant"

echo "all checks passed"
