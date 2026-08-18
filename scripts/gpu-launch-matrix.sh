#!/usr/bin/env bash
# Execute staged case runners across a launch matrix. Runs ON THE GPU HOST,
# inside a cuda-oxide checkout, over runner crates copied into its examples
# directory as `case_<id>`.
#
# Two lessons from the Stage-0 probe are wired in rather than remembered:
#
#   1. Build and execution are timed and bounded SEPARATELY. v1 of that probe
#      wrapped the whole `cargo oxide run` in a short watchdog, so the watchdog
#      killed dependency compilation and reported 24/24 "watchdog-fired" —
#      including for the case that was safe by construction.
#   2. A killed run is recorded as `watchdog-fired`, never as `deadlock`. On
#      sm_86 a divergent barrier *completes*; wording that presumes a hang
#      invents two dozen findings that are not there.
#
# Output is one stdout file and one sanitizer file per (case, block), which is
# exactly what `simt-diff ingest` reads. Nothing here classifies anything: the
# controller does that, from the records.
#
# Usage (on the box, from the cuda-oxide checkout root):
#   scripts/gpu-launch-matrix.sh [OUT_DIR]
# Env:
#   BLOCKS          block sizes to run            (default "32 64 128")
#   EXAMPLES        where the staged runners are  (default crates/rustc-codegen-cuda/examples)
#   RUN_WATCHDOG    seconds per execution         (default 20)
#   BUILD_WATCHDOG  seconds per build             (default 1800)
set -uo pipefail

OUT=${1:-launch-matrix-results}
BLOCKS=${BLOCKS:-"32 64 128"}
EXAMPLES=${EXAMPLES:-crates/rustc-codegen-cuda/examples}
RUN_WATCHDOG=${RUN_WATCHDOG:-20}
BUILD_WATCHDOG=${BUILD_WATCHDOG:-1800}

mkdir -p "$OUT/logs"
TSV="$OUT/results.tsv"; : > "$TSV"
printf 'case\tblock\tmode\texit\tseconds\toutcome\tnote\n' >> "$TSV"

{
  echo "== date =="; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== gpu =="; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
  echo "== nvcc =="; nvcc --version 2>&1 | tail -2
  echo "== compute-sanitizer =="; compute-sanitizer --version 2>&1 | head -3
  echo "== rustc =="; rustc --version
  # The box's .git can be stale relative to the synced worktree, so the
  # authoritative cuda-oxide revision is recorded on the CONTROLLER side.
  echo "== cuda-oxide (box .git, may be stale) =="; git log -1 --format='%H %ad' --date=short 2>&1
} > "$OUT/environment.txt" 2>&1

CASES=$(ls -d "$EXAMPLES"/case_* 2>/dev/null | xargs -n1 basename 2>/dev/null)
if [ -z "$CASES" ]; then
  echo "no staged runners in $EXAMPLES (expected directories named case_<id>)" >&2
  exit 2
fi
echo "staged cases: $(echo "$CASES" | wc -w | tr -d ' ')"

echo "### phase 1: build (watchdog ${BUILD_WATCHDOG}s, generous on purpose)"
for c in $CASES; do
  log="$OUT/logs/${c}-build.log"
  start=$(date +%s)
  timeout "$BUILD_WATCHDOG" cargo oxide build "$c" > "$log" 2>&1
  st=$?; secs=$(( $(date +%s) - start ))
  bin=$(find "$EXAMPLES/$c/target" -maxdepth 2 -type f -name "$c" -perm -u+x 2>/dev/null | head -1)
  if [ -n "$bin" ]; then
    echo "  built  $c (${secs}s)"
    printf '%s\t-\tbuild\t%s\t%s\tbuilt\t%s\n' "$c" "$st" "$secs" "$bin" >> "$TSV"
  else
    echo "  FAILED $c (${secs}s) -- see $log"
    printf '%s\t-\tbuild\t%s\t%s\tcompile-failed\t%s\n' "$c" "$st" "$secs" "$log" >> "$TSV"
  fi
done

echo "### phase 2: execute (watchdog ${RUN_WATCHDOG}s per run)"
for c in $CASES; do
  bin=$(find "$EXAMPLES/$c/target" -maxdepth 2 -type f -name "$c" -perm -u+x 2>/dev/null | head -1)
  [ -z "$bin" ] && continue
  for b in $BLOCKS; do
    raw="$OUT/${c}-block${b}.stdout"
    start=$(date +%s)
    timeout --kill-after=5 "$RUN_WATCHDOG" "$bin" "$b" > "$raw" 2>&1
    st=$?; secs=$(( $(date +%s) - start ))
    case $st in
      0)   outcome=completed ;;
      124|137) outcome=watchdog-fired ;;
      *)   outcome=nonzero-exit ;;
    esac
    printf '%s\t%s\traw\t%s\t%s\t%s\t%s\n' "$c" "$b" "$st" "$secs" "$outcome" "$raw" >> "$TSV"
    echo "  $c block=$b -> $outcome (${secs}s)"

    san="$OUT/${c}-block${b}.sanitizer"
    timeout --kill-after=5 $(( RUN_WATCHDOG * 3 )) \
      compute-sanitizer --tool synccheck "$bin" "$b" > "$san" 2>&1
    sst=$?
    case $sst in
      0)   souts=completed ;;
      124|137) souts=watchdog-fired ;;
      *)   souts=nonzero-exit ;;
    esac
    printf '%s\t%s\tsynccheck\t%s\t-\t%s\t%s\n' "$c" "$b" "$sst" "$souts" "$san" >> "$TSV"
  done
done

echo
echo "results:     $TSV"
echo "environment: $OUT/environment.txt"
echo "ingest these on the controller; nothing here draws a conclusion."
