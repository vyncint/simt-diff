# simt-diff — SIMT Differential Laboratory

**Status: Stage 0 (reconnaissance).** No implementation yet, deliberately.

A differential laboratory for SIMT static analyzers. It generates kernels
whose convergence property is known *by construction*, analyzes them with
[reconverge](https://github.com/vyncint/reconverge), executes them on a real
NVIDIA GPU under a controlled launch matrix, cross-checks against NVIDIA
Compute Sanitizer, and classifies the agreement between those independent
evidence sources — then minimizes anything interesting into a reproducer.

## What it is not

- not a CUDA compiler fuzzer (CUDAsmith)
- not a GPU application fuzzer (cuFuzz)
- not a duplicate of cuda-oxide's own CPU-vs-GPU MIR fuzzer
- not a profiler, not a simulator, not a fork of reconverge's analyzer

## Read this first

[`docs/research-baseline.md`](docs/research-baseline.md) — the Stage 0
deliverable: what the upstream projects actually do today, which parts of
this brief already exist upstream, and the measured hardware facts the
design depends on.

Two findings from it govern everything else:

1. reconverge already ships a mutation corpus with published
   precision/recall, and two hardware sessions that are **prepared but
   never run**. This project is best positioned as the generalization of
   its session #2, not as an unrelated system.
2. A clean GPU run proves nothing universal, and a `warning`-tier finding
   is not a false positive. The classification model encodes both.
