#!/usr/bin/env bash
set -euo pipefail

original="${1:?usage: ROLLBACK.sh <original-file> <target-file>}"
target="${2:?usage: ROLLBACK.sh <original-file> <target-file>}"

cp -- "$original" "$target"
original_hash="$(sha256sum "$original" | awk '{print $1}')"
restored_hash="$(sha256sum "$target" | awk '{print $1}')"
test "$original_hash" = "$restored_hash"
printf 'ROLLBACK_RESULT=PASS\n'
printf 'ORIGINAL_SHA256=%s\n' "$original_hash"
printf 'RESTORED_SHA256=%s\n' "$restored_hash"
printf 'RESTORED_BEHAVIOR=baseline discrete CityUAV environment source restored byte-for-byte\n'

