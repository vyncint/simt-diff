#!/usr/bin/env bash
# Controller side of the launch matrix: turn the GPU host's output files into
# records, then classify.
#
# Kept separate from the box script on purpose. Execution and interpretation
# happen on different machines, and the provenance of every dynamic observation
# is recorded with it rather than assumed.
#
# Usage:
#   scripts/ingest-launch-matrix.sh <results-dir> <cases-dir> [provenance]
#
# <results-dir>  what gpu-launch-matrix.sh wrote (synced back from the box)
# <cases-dir>    the case directories those runners were generated from
set -euo pipefail

RESULTS=${1:?usage: $0 <results-dir> <cases-dir> [provenance]}
CASES=${2:?usage: $0 <results-dir> <cases-dir> [provenance]}
PROVENANCE=${3:-"$RESULTS on an unnamed GPU host"}

shopt -s nullglob
found=0
for raw in "$RESULTS"/case_*-block*.stdout; do
  base=$(basename "$raw")
  id=${base#case_}; id=${id%%-block*}
  block=${base##*-block}; block=${block%.stdout}
  dir="$CASES/$id"
  if [ ! -d "$dir" ]; then
    echo "skip $base: no case directory $dir" >&2
    continue
  fi
  san="$RESULTS/case_${id}-block${block}.sanitizer"
  args=(ingest "$dir" --stdout "$raw" --block "$block" --provenance "$PROVENANCE")
  [ -f "$san" ] && args+=(--sanitizer "$san")
  cargo run --quiet -p simt-diff -- "${args[@]}"
  found=$((found + 1))
done

if [ "$found" -eq 0 ]; then
  echo "no runner output found in $RESULTS" >&2
  exit 2
fi
echo
echo "ingested $found run(s); classifying"
for dir in "$CASES"/*/; do
  [ -f "$dir/analyzer.json" ] || continue
  cargo run --quiet -p simt-diff -- compare "$dir" || true
done
