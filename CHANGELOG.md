# Changelog

All notable changes to this project are documented here, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every entry reports **what was refuted as well as what was found**, deliberately.
A laboratory that only publishes its confirmations is publishing its selection
bias; the versions of a rule that turned out to be wrong are the part a reader
can learn the most from.

## [Unreleased]

Nothing yet.

## [0.1.0] — 2026-08-18

First public version. Five stages, all measured against reconverge 0.1.6 on the
pinned nightly; no GPU was used after Stage 1.

### Added

- **Stage 0** — [`docs/research-baseline.md`](docs/research-baseline.md).
  Measured reconnaissance on an A10G (sm_86) that overturned three assumptions
  the project was designed around: a divergent `sync_threads()` *completes*,
  `synccheck` reports nothing for it, and an invalid warp mask returns a value
  byte-identical to the valid case.
- **Stage 1** — one vertical slice end to end: generate, analyze with real
  reconverge, execute on real hardware, classify.
- **Stage 3** — fourteen hand-written templates, each predicting what the
  analyzer's documentation says should happen. Fourteen predictions held.
- **Stage 4** — the mutation engine: a closed kernel IR, a 32-lane exact
  interpreter that *computes* each mutant's oracle rather than inheriting it,
  and a rule-based model of reconverge in which every prediction carries its
  provenance (quoted / extrapolated / measured).
- **Stage 5** — a minimizer that preserves oracle *and* observation, reproducer
  packaging with a self-contained `verify.sh`, and a regression corpus that
  separates generator drift from analyzer drift.

### Found

Five places where reconverge's behaviour and its documentation differ, three of
them because the tool is better than documented — and, across 246 generated
kernels, **no false positive at a gating tier**, including cases built
specifically to force one. See [`docs/stage-4.md`](docs/stage-4.md).

### Refuted

The finding about witness promotion in multi-site kernels was stated wrongly
four times before the fifth version survived. Two of those were killed by cases
built to demonstrate them, one by a held-out corpus. All five versions and their
refutations are recorded rather than tidied away.
