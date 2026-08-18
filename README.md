# simt-diff — SIMT Differential Laboratory

A differential laboratory for SIMT static analyzers. It generates kernels whose
convergence property is known *by construction*, analyzes them with
[reconverge](https://github.com/vyncint/reconverge), executes them on a real
NVIDIA GPU, cross-checks against NVIDIA Compute Sanitizer, and classifies the
agreement between those independent evidence sources.

## What it is not

- not a CUDA compiler fuzzer (CUDAsmith)
- not a GPU application fuzzer (cuFuzz)
- not a duplicate of cuda-oxide's own CPU-vs-GPU MIR fuzzer
- not a profiler, not a simulator, not a fork of reconverge's analyzer

## Status

| stage | what it delivered |
|---|---|
| 0 | [`docs/research-baseline.md`](docs/research-baseline.md) — measured reconnaissance; three of the brief's assumptions overturned |
| 1 | [`docs/stage-1.md`](docs/stage-1.md) — one vertical slice end to end: real analyzer, real A10G |
| 3 | [`docs/conformance-reconverge-0.1.6.md`](docs/conformance-reconverge-0.1.6.md) — fourteen hand-written templates, fourteen predictions held |
| 4 | [`docs/stage-4.md`](docs/stage-4.md) — the mutation engine: oracles computed, not inherited |

Not built yet: the minimizer, reproducer packaging, the regression corpus, issue
drafting, device-buffer tracing, and the launch matrix beyond one block size.

## Using it

The static half needs no GPU — reconverge builds and runs on a laptop, and every
number in the Stage 4 document was produced without one.

```sh
simt-diff doctor                  # which stages this host can run
simt-diff templates               # the hand-written semantic templates
simt-diff mutate --depth 1        # the generated corpus, with oracle + prediction
simt-diff mutate --source <id>    # one generated kernel, as analyzed
simt-diff generate <template>     # write one case directory
simt-diff analyze <case-dir>      # run reconverge, record findings
simt-diff ingest <case-dir> …     # record a GPU run performed on another host
simt-diff compare <case-dir>      # classify from whatever records exist
simt-diff conformance --mutants   # sweep the corpus: predict, analyze, classify
```

`cargo-reconverge` is found on `PATH`, via `$SIMT_DIFF_RECONVERGE`, or with
`--reconverge`. Exit codes: 0 ok, 1 something wants a human, 2 tool error.

## The two rules that shape everything

Both were learned the expensive way and are enforced in code, not documentation:

1. **A clean GPU run never argues against a static finding.** A divergent
   `sync_threads()` completes on sm_86 with the barrier provably still inside the
   branch, and `synccheck` says nothing (baseline §9.3, §9.4). A laboratory that
   read completion as safety would call reconverge's flagship diagnostic a false
   positive, on hardware, repeatably, and be wrong every time.
2. **Only a gating-tier finding can be a false positive.** A `warning` is not an
   assertion. Reporting one would be reporting the tool for working as specified.

## Reading a claim from this repository

Every prediction carries its provenance, because "reconverge is wrong" and "this
model is wrong" are different claims and the difference is not always obvious in
advance:

- **quoted** — the documentation states this behaviour. A violation is about
  reconverge.
- **extrapolated** — inferred from a documented *reason*. A violation is about
  this model first.
- **measured** — undocumented, established by a run this repository recorded, with
  the case named. A violation is a regression.
