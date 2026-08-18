# Stage 6 — the launch matrix

Every rule in the model was written at `grid=1, block=32`. Block size is the one
dimension none of them was fitted to, and it is the dimension where the
*construction oracle* genuinely moves:

```rust
if warp::warp_id() == 0 {
    thread::sync_threads();
}
```

At 32 threads this is one warp, the guard is true for every thread, and nothing
diverges — the kernel is **safe**. At 64 it is two warps, one of them skips a
block-wide barrier, and the kernel is **undefined**. Same source, opposite
labels, decided entirely by the launch declared in the contract.

That makes the matrix a real held-out test rather than another sweep: the oracle
changes, the prediction does not, so a prediction violation would be a genuine
finding and a clean sheet is genuine evidence.

Everything below is static. Executing multi-warp kernels needs a GPU, and that
half is staged but not yet run — see "The dynamic half" at the end.

## Result

The same 147-case corpus, at three block sizes, against reconverge 0.1.6:

| launch | predictions held | violated | cases wanting a human |
|---|---:|---:|---:|
| `block=32` | 147 | 0 | 0 |
| `block=64` | 147 | 0 | 0 |
| `block=128` | 147 | 0 | 0 |

Comparing the three sweeps case by case, over the 147 cases present in all of
them:

- **11 cases changed their construction oracle** between 32 and 64 —
  `KNOWN_SAFE` → `KNOWN_UNSAFE` — and the same 11 at 128. Every one is a kernel
  whose divergence comes from `warp_id()`: the seed itself, and mutants of
  `barrier_divergent_intra_warp`, `barrier_divergent_loop_break` and
  `barrier_in_helper_divergent_call` that swapped their guard onto it.
- **0 cases changed the analyzer's answer, at any block size.**
- 11 classifications changed as a consequence, from `AgreementSafe` — a warning
  on a safe kernel, which is within specification — to `AnalyzerUnsupported`:
  the construct was seen and deliberately not promoted, which is the documented
  behaviour for a lane-environment guard.

The model predicted every one of those rows correctly at a launch it had never
seen, which is the strongest evidence so far that the rules describe reconverge
rather than describing the corpus they were fitted to.

## What the invariance costs

reconverge's answer to that kernel is `RC001/warning`, with no witness, at every
block size. At 32 that is exactly right — the program is safe, and a *gating*
finding would have been a false positive. At 64 the program is undefined, and the
same non-gating answer means **CI does not stop it**.

Nothing here is a contradiction: `## Limitations` says guards built on the
lane-environment registers stay warnings, and that is what happens. What the
matrix adds is the price. The limitation is not "imprecision at the margins";
at block ≥ 64 it is a deadlock-class bug that the gate lets through, and the
kernel that exhibits it is four lines long.

It also shows the information is *there*. `#[launch_contract(block = (64,1,1))]`
declares the shape, `warp_id()` is the thread index divided by 32, and the
witness interpreter already replays lanes and evaluates thread-index arithmetic.
What it cannot do is see across warps: a 32-lane replay contains exactly one
warp, so a whole-warp divergence has no divergent lane pair *within* the replay.
That is a design boundary rather than a missing intrinsic, which is why it is
filed as an enhancement rather than a bug.

## The dynamic half

Not run yet. It needs a GPU, and it is the only part of this laboratory that
does.

The interesting measurement is not the divergent barrier — the baseline already
measured that completing on sm_86 with `synccheck` silent
(`docs/research-baseline.md` §9.3–§9.4), and an undefined program has no
reference model to check against. It is the **collectives**, where multi-warp
launches produce a per-warp value prediction that a single warp cannot:

```rust
if warp::warp_id() == 0 {
    b = warp::ballot_sync(0xffff_ffff, true);
}
```

At `block=64` the interpreter predicts `0xffffffff` for lanes 0–31 and `0` for
lanes 32–63, because warp 1 never reaches the call and keeps its initial value.
Masks are warp-local, and that prediction is a claim about *this laboratory's*
warp model as much as about the hardware. If the values come back otherwise, the
interpreter is wrong and every multi-warp oracle above is suspect.

Two scripts stage it, deliberately split across the two machines:

- `scripts/gpu-launch-matrix.sh` runs on the GPU host inside a cuda-oxide
  checkout, builds each staged runner and executes it at each block size, raw
  and under `compute-sanitizer --tool synccheck`. It draws no conclusions; it
  writes files.
- `scripts/ingest-launch-matrix.sh` runs on the controller, turns those files
  into records with their provenance attached, and classifies.

Both carry the Stage-0 lessons in code rather than in a comment: build and
execution are bounded separately, because a watchdog around compilation once
produced 24 fictitious findings; and a killed run is recorded as
`watchdog-fired`, never as `deadlock`, because on this hardware a divergent
barrier completes.

## What this does not establish

- One analyzer version, one grid (`grid=1`), one dimension of the launch. Grid
  size, block dimensions y and z, and shared memory are all unexplored.
- Three block sizes are three points, not a function. 96 threads, or a block
  that is not a multiple of the warp size, would be the interesting next probes —
  a partial final warp is a shape none of these cover.
- Nothing about hardware. Every number above came from a laptop.
