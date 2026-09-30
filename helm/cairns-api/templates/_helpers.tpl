{{/* Standard name helpers. */}}
{{- define "cairns-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "cairns-api.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "cairns-api.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "cairns-api.labels" -}}
app.kubernetes.io/name: {{ include "cairns-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "cairns-api.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cairns-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Subpath the app is mounted under. Falls back to ingress.path; "/" normalises to
"" because FastAPI's root_path must be empty (not "/") when served at the root.
*/}}
{{- define "cairns-api.rootPath" -}}
{{- $p := default .Values.ingress.path .Values.rootPath -}}
{{- if or (not $p) (eq $p "/") -}}{{- "" -}}{{- else -}}{{- printf "/%s" (trimAll "/" $p) -}}{{- end -}}
{{- end -}}

{{/* PVC name: an existing claim wins, otherwise the one this chart creates. */}}
{{- define "cairns-api.pvcName" -}}
{{- default (printf "%s-data" (include "cairns-api.fullname" .)) .Values.persistence.existingClaim -}}
{{- end -}}

{{/*
Qdrant URL: the bundled subchart's Service when enabled, else the external one.
The subchart names its Service <release>-qdrant unless fullnameOverride is set.
*/}}
{{- define "cairns-api.qdrantUrl" -}}
{{- if .Values.qdrant.enabled -}}
{{- $svc := default (printf "%s-qdrant" .Release.Name) .Values.qdrant.fullnameOverride -}}
{{- printf "http://%s:6333" $svc -}}
{{- else -}}
{{- trimSuffix "/" .Values.vectorStore.url -}}
{{- end -}}
{{- end -}}
