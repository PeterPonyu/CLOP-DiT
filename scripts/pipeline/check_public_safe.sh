#!/usr/bin/env bash
# Pre-push guard: refuse to push if the set of changed files contains
# paths that leak the private revision manuscript, response letter, or
# internal revision logs.
#
# Install:
#     ln -s ../../scripts/pipeline/check_public_safe.sh .git/hooks/pre-push
#
# Hooks receive the remote name and URL on argv, and the list of ref
# updates on stdin. For each ref update, we diff the pushed range
# against its remote counterpart and scan for blocked paths.

set -euo pipefail

BLOCK_RE='^(configs/pipeline\.yaml$|revision/manuscripts/|revision/figure_fix_reports/.*\.md$|revision/response_letter/|revision/REVISION_LOG\.md$|revision/REVIEWER_RESPONSE_SYNTHESIS\.md$|revision/reviewer_response_draft\.md$|revision/rebuttal_crossref_audit\.md$|revision/LEAKAGE_REAUDIT\.md$|revision/VCD_AUDIT_FINDINGS_.*\.md$)'

remote_name="${1:-origin}"
remote_url="${2:-}"

# Only enforce when pushing to the public origin. Private forks can opt in
# by exporting CLOPDIT_LEAK_GATE_ALWAYS=1.
if [ "${CLOPDIT_LEAK_GATE_ALWAYS:-0}" != "1" ]; then
  case "$remote_url" in
    *PeterPonyu/CLOP-DiT*) : ;;   # enforce
    *) exit 0 ;;
  esac
fi

zero="0000000000000000000000000000000000000000"
fail=0

while read -r local_ref local_sha remote_ref remote_sha; do
  if [ -z "${local_sha:-}" ] || [ "$local_sha" = "$zero" ]; then
    # Deletion push — nothing to scan.
    continue
  fi
  if [ "${remote_sha:-$zero}" = "$zero" ]; then
    range="$local_sha"
  else
    range="${remote_sha}..${local_sha}"
  fi

  leaked=$(git diff --name-only "$range" | grep -E "$BLOCK_RE" || true)
  if [ -n "$leaked" ]; then
    fail=1
    printf 'LEAK-GATE: refusing to push %s → %s\n' "$local_ref" "$remote_ref"
    printf 'blocked paths in range %s:\n' "$range"
    printf '%s\n' "$leaked" | sed 's/^/  /'
  fi
done

if [ "$fail" -ne 0 ]; then
  printf '\nUnstage or drop the blocked paths, or set CLOPDIT_LEAK_GATE_ALWAYS=0\n'
  printf 'and push to a private remote if the content is intentional.\n'
  exit 1
fi

exit 0
