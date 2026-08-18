#!/usr/bin/env bash
# Stage-0 feasibility probe for the SIMT Differential Laboratory.  v2.
#
# v1 was wrong in a way worth recording: it wrapped `cargo oxide run` in the
# watchdog, so the watchdog killed dependency COMPILATION and reported 24/24
# "watchdog-fired" -- including for the KNOWN_SAFE probe. Build and execution
# must be separately timed, and only execution may carry a short watchdog.
#
# Phase 1: build every probe, no watchdog, failures recorded.
# Phase 2: execute the built binary under a short watchdog, raw and under
#          `compute-sanitizer --tool synccheck` (the same invocation
#          cargo-oxide's own `sanitize` builds: `--tool <t>` then the binary).
set -uo pipefail

OUT=${1:-probe-results}
RUN_WATCHDOG=${RUN_WATCHDOG:-20}
BUILD_WATCHDOG=${BUILD_WATCHDOG:-1800}
BLOCKS=${BLOCKS:-"32 64 128"}
PROBES="probe_safe_barrier probe_divergent_barrier probe_mask_full probe_mask_shrunk"
EX=crates/rustc-codegen-cuda/examples

mkdir -p "$OUT/logs"
TSV="$OUT/results.tsv"; : > "$TSV"
printf 'probe\tblock\tmode\texit\tseconds\tverdict\tnote\n' >> "$TSV"
BUILD="$OUT/build.tsv"; : > "$BUILD"
printf 'probe\texit\tseconds\tbinary\n' >> "$BUILD"

{
  echo "== date =="; date -u +%Y-%m-%dT%H:%M:%SZ
  echo "== gpu =="; nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
  echo "== nvcc =="; nvcc --version 2>&1 | tail -2
  echo "== compute-sanitizer =="; compute-sanitizer --version 2>&1 | head -3
  echo "== rustc =="; rustc --version
  # NOTE: the box's .git can be stale relative to the synced worktree, so the
  # authoritative cuda-oxide revision is recorded on the CONTROLLER side.
  echo "== cuda-oxide (box .git, may be stale) =="; git log -1 --format='%H %ad' --date=short 2>&1
} > "$OUT/environment.txt" 2>&1

echo "### phase 1: build (watchdog ${BUILD_WATCHDOG}s, generous on purpose)"
for probe in $PROBES; do
  log="$OUT/logs/${probe}-build.log"
  start=$(date +%s)
  timeout "$BUILD_WATCHDOG" cargo oxide build "$probe" > "$log" 2>&1
  st=$?; secs=$(( $(date +%s) - start ))
  bin=$(find "$EX/$probe/target" -maxdepth 2 -type f -name "$probe" -perm -u+x 2>/dev/null | head -1)
  printf '%s\t%s\t%s\t%s\n' "$probe" "$st" "$secs" "${bin:-NONE}" >> "$BUILD"
  echo "  build $probe -> exit $st (${secs}s) bin=${bin:-NONE}"
done

echo "### phase 2: execute (watchdog ${RUN_WATCHDOG}s)"
run_one() { # probe block mode cmd...
  local probe=$1 block=$2 mode=$3; shift 3
  local log="$OUT/logs/${probe}-b${block}-${mode}.log"
  local start st secs verdict note
  start=$(date +%s)
  timeout --signal=TERM --kill-after=5 "$RUN_WATCHDOG" "$@" > "$log" 2>&1
  st=$?; secs=$(( $(date +%s) - start ))
  case $st in
    0)   verdict="completed" ;;
    124|137) verdict="watchdog-fired" ;;   # evidence only; never "deadlock"
    *)   verdict="nonzero-exit" ;;
  esac
  note=$(grep -m1 -oE 'RESULT .*|Barrier error[^"]*|barrier.*divergen[^"]*|========= [A-Z].*' "$log" 2>/dev/null | head -1 | cut -c1-90)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$probe" "$block" "$mode" "$st" "$secs" "$verdict" "${note:-}" >> "$TSV"
  echo "  [$mode] $probe b=$block -> $verdict (exit $st, ${secs}s) ${note:-}"
}

while IFS=$'\t' read -r probe st secs bin; do
  [ "$probe" = "probe" ] && continue
  if [ "$bin" = "NONE" ] || [ ! -x "$bin" ]; then
    echo "  SKIP $probe: no binary (build exit $st)"
    printf '%s\t-\t-\t-\t-\tbuild-failed\tsee logs\n' "$probe" >> "$TSV"
    continue
  fi
  for block in $BLOCKS; do
    run_one "$probe" "$block" raw       "$bin" "$block"
    run_one "$probe" "$block" synccheck compute-sanitizer --tool synccheck "$bin" "$block"
  done
done < "$BUILD"

echo; echo "== build =="; cat "$BUILD"; echo; echo "== results =="; cat "$TSV"
