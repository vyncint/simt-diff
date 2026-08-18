#!/bin/sh
# Re-run the analyzer on this case and compare against expected.json.
#
# Exit 0: the observation is unchanged.
# Exit 1: it moved -- which is the interesting outcome, and why this script
#         parses the analyzer's JSON instead of grepping it.
# Exit 2: the check could not run.
set -eu
cd "$(dirname "$0")"

RECONVERGE="${SIMT_DIFF_RECONVERGE:-cargo-reconverge}"
command -v "$RECONVERGE" >/dev/null 2>&1 || {
    echo "verify: $RECONVERGE not on PATH; set SIMT_DIFF_RECONVERGE" >&2
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    echo "verify: python3 is required to read the analyzer's JSON" >&2
    exit 2
}

rm -rf kernel/target/reconverge
( cd kernel && "$RECONVERGE" reconverge check --message-format json --strict ) > .verify-out.json 2>.verify-err.txt || true
WITNESSES=$(ls kernel/target/reconverge/witness-*.json 2>/dev/null | wc -l | tr -d ' ')

python3 - "$WITNESSES" <<'PY'
import json, sys

witnesses = int(sys.argv[1])
expected = json.load(open("expected.json"))

found = []
for line in open(".verify-out.json"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        doc = json.loads(line)
    except ValueError:
        continue
    if doc.get("schema") != "findings.v1":
        continue
    for f in doc.get("findings", []):
        if f.get("code") in ("RC001", "RC002"):
            found.append(f"{f['code']}/{f['confidence']}".lower())

signature = f"{','.join(sorted(found)) or 'silent'}|{witnesses}w"
want = expected["signature"]
print(f"expected: {want}")
print(f"observed: {signature}")
if signature == want:
    print("OK: the observation is unchanged")
    sys.exit(0)
print("MOVED: this case no longer reproduces what it was packaged for", file=sys.stderr)
print("       See README.md for what the expectation rests on.", file=sys.stderr)
sys.exit(1)
PY
