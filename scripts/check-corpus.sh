#!/usr/bin/env bash
# check-corpus.sh — generator-drift gate (docs/stage-5.md).
#
# Every regression-corpus entry stores a *recipe* — a seed template plus the
# mutation operators applied to it — and the SHA-256 of the kernel that recipe
# produced. This gate rebuilds each entry and compares the hash.
#
# It deliberately does NOT run the analyzer, so it needs no pinned nightly and
# no cargo-reconverge: it catches the failure this repository is responsible for
# (the generator changed and a case silently became a different case), not the
# one reconverge is responsible for (the same program, a different answer).
# That second check is `just regress`, which needs the analyzer and therefore
# runs locally rather than in CI.
set -euo pipefail

cd "$(dirname "$0")/.."
cargo run --quiet -p simt-diff -- corpus check "$@"
