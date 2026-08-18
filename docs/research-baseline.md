# Research baseline — Stage 0

Status: **in progress.** Everything below was read out of the checked-out
sources on 2026-08-18, not recalled. Claims that could not be verified
without hardware are marked `MEASURE` and are answered by the probe in
[§9](#9-open-empirical-questions).

Versions this document describes:

| Component | Version / commit | How established |
|---|---|---|
| reconverge | `0.1.6`, `46f64cc` (2026-08-18) | `Cargo.toml` workspace version, `git log -1` |
| cuda-oxide (upstream main) | `50d07314` (2026-08-18) | `git log -1 upstream/main` |
| cuda-oxide (reconverge's pin) | `a766fc26` | `conformance/PIN` |
| Rust toolchain | `nightly-2026-04-03` | `rust-toolchain.toml`, both repos |

The two cuda-oxide revisions differ. reconverge's conformance suite is
pinned; a laboratory that generates *new* kernels should track upstream
main and record which revision it built against, per case.

---

## 1. What this project is, in one paragraph

An external validation layer around a SIMT static analyzer. It generates
kernels whose SIMT property is known **by construction**, analyzes them
with reconverge, executes them on a real NVIDIA GPU under controlled
launch configurations, collects dynamic evidence, and classifies the
agreement between those independent sources. Its output is not a pass
count; it is a small, minimized, reproducible counterexample with an
explicit statement of what the evidence does and does not prove.

## 2. reconverge as it actually is at 0.1.6

### 2.1 Diagnostics and tiers

| Code | Tier | What it means |
|---|---|---|
| `RC001` | confirmed / warning | an all-threads barrier reachable under thread-divergent control |
| `RC002` | confirmed / warning | warp collective at a non-convergent point, or a mask naming absent lanes |
| `RC003` | deny | `&mut [T]` as a `#[kernel]` parameter |
| `RC004` | deny | static shared memory over the target's limit |
| `RC005` | warning | launch-contract inconsistency |

`RC006`/`RC007` are reserved for v1.1 and do not exist yet.

The confidence ladder is load-bearing and must be preserved in every
record this laboratory writes:

- **deny** — proven from syntax alone. Always shown, gates CI.
- **confirmed** — the 32-lane witness interpreter replayed a concrete
  hang under a concrete launch. Always shown, gates CI.
- **warning** — hidden unless `--strict`, never gates.

> "Zero false positives at default confidence is a requirement, not a
> goal." — README

That sentence sets the bar for any false-positive claim this laboratory
makes: a `warning`-tier finding is *not* a false positive, because the
tool does not assert it at gating confidence. Only a `deny` or
`confirmed` finding can be one.

### 2.2 The stable machine-readable interface

Verified against `crates/cargo-reconverge/src/main.rs` and `schemas/`:

| Interface | Contract |
|---|---|
| `cargo reconverge check --message-format json` | one `findings.v1` document per analyzed crate, **one per line** (JSONL) |
| `cargo reconverge check --sarif <PATH>` | SARIF 2.1.0 |
| exit code | `0` clean · `1` findings at deny/confirmed · `2` tool error |
| `<target>/reconverge/findings-*.json` | `findings.v1` per crate |
| `<target>/reconverge/witness-<crate>-<kernel>-*.json` | `witness.v1`, written **only for witness-confirmed findings** |
| `<target>/reconverge/unimap-*.json` | `unimap.v1` |

`findings.v1` fields: `code`, `confidence` (`deny`/`confirmed`/`warning`),
`message`, `kernel`, `span`, `notes[]`, `help`, `explain`, `provenance[]`.
Schemas are additive-only within v1 and consumers must tolerate unknown
fields — so the adapter must deserialize permissively and keep raw bytes.

`witness.v1` carries what a differential record needs and `findings.v1`
does not: `launch { grid, block, warp }`, `lanes`, delta-encoded
`steps[]`, and `verdict.kind` ∈ {`hang`, `undefined-behavior`,
`completed`, `no-witness`}.

**Consequence for the adapter:** a complete static record needs both the
JSONL on stdout *and* the witness files on disk. Neither alone is enough.

### 2.3 The analyzed surface, exactly

From `crates/reconverge-dialect-oxide/src/simt.rs` — this is the
generator's target vocabulary, and generating outside it produces
`AnalyzerUnsupported`, not a bug:

- **Barriers (`CallKind::Barrier`)** — `sync_threads`, `cluster_sync`,
  and `grid::sync`. The mbarrier family (`barrier::Barrier`) is
  deliberately excluded: it is a phase-counted split barrier where
  partial participation is the designed use.
- **Warp collectives (`CallKind::WarpCollective`)** — the masked
  `*_sync` surface: `ballot/any/all_sync`, `shuffle{,_up,_down,_xor}_sync`,
  `match_{any,all}_sync`, `redux_sync_{add,and,or,xor}`,
  `elect_sync`, `is_elected_sync`, `sync_mask`.
- **Divergent environment reads** — `active_mask`, `lanemask_{lt,le,eq,ge,gt}`,
  `warp_id`, `live_lanes_1d`.
- **Index witnesses** — `index_1d`, `index_2d`, `index_2d_runtime`,
  `warp_index`, plus `threadIdx_{x,y,z}` and `lane_id`.
- **Explicitly outside v1** — the unmasked convenience wrappers
  `warp::shuffle`, `warp::ballot`, `warp::all/any`, the `reduce_*`
  helpers. `simt.rs` has a test named
  `unmasked_wrappers_are_the_documented_v1_gap` asserting this.

### 2.4 Documented limitations — the research targets

These are quoted from the README because they define where a discrepancy
is *expected* rather than interesting:

1. No SMT; uniformity is dataflow. Data races are out of scope entirely.
2. Irreducible CFGs degrade to all-divergent for that function.
3. Interprocedural analysis is summary-based: per-function
   `may_contain_barrier` / `may_contain_warp_op` bits, no context
   sensitivity. **Call-site findings stay at `warning` and are never
   witness-promoted.**
4. Lane-environment guards stay warnings — the interpreter cannot yet
   evaluate `lanemask_*` / `warp_id` / `live_lanes_1d` values, which
   needs width-typed evaluation of integer `!` and truncating casts.
5. Non-literal masks (a named `const`, or anything computed) cannot be
   evaluated through `rustc_public` at the pin, so RC002 reports
   convergence and says the mask was not evaluable.
6. Opaque regions (`asm!`, unmodeled intrinsics) are counted and
   reported, not guessed at.

Items 3, 4 and 5 are the highest-value generator targets, because each
predicts a *specific, checkable* degradation — not "maybe a bug", but
"this construct must land at warning tier, and never be confirmed".
A case that violates one of those predictions is interesting in either
direction.

---

## 3. What already exists — and must not be rebuilt

This is the section that most changes the project's shape. Three of the
things the brief proposes already exist upstream, in whole or in part.

### 3.1 reconverge already has a mutation corpus

`conformance/MUTATION.md`, regenerated by CI and diffed against the
committed copy. Five operators over the pinned upstream examples:

| class | injected bug | expected | mutants | detected (default) | detected (`--strict`) |
|---|---|---|---:|---:|---:|
| wrapbar | barrier wrapped in an index-derived `if` | RC001 | 67 | 47/67 (70%) | 62/67 (92%) |
| delbar | barrier deleted (data race) | — | 67 | 0/67 | 0/67 |
| wrapcol | collective wrapped the same way | RC002 | 14 | 0/14 | 14/14 (100%) |
| shrinkmask | full mask shrunk to `0x0000_ffff` | RC002 | 17 | 0/17 | 0/17 |
| mutslice | `DisjointSlice<T>` → `&mut [T]` | RC003 | 414 | 393/393 | 393/393 |

Precision at default confidence: **1.000**.

So "mutate a kernel and see whether reconverge notices" is done. Two
rows are worth staring at, because they are published gaps rather than
unknowns:

- **`shrinkmask` at 0% by design.** "a shrunk full mask at a *convergent*
  call site is a mask-lane mismatch only under launch shapes the static
  engine does not model (RC002 v1 checks convergence; it does not do mask
  arithmetic against launch shapes)."
- **`delbar` at 0% by design.** A deleted barrier is a race, outside the
  decidable slice.

Neither is a bug to be discovered. But **neither has ever been run on
hardware**, so what the GPU actually does for those mutants is unknown —
and that is exactly the evidence this laboratory can add.

### 3.2 reconverge already has two hardware sessions — prepared, never run

`docs/hardware/session-1.md` and `session-2.md`, both marked
**"prepared, not run"**, both blocked on the same thing: someone with a
GPU host. Their stated goals:

- **Session #1 — witness calibration.** For each true-positive kernel,
  record what actually happens per compute capability: hang,
  wrong-result, or accidental pass. The purpose is to keep the witness
  verdict wording honest — the tool says "undefined behavior, usually
  hangs" and this session is what turns "usually" into recorded data.
- **Session #2 — sanitizer cross-check.** Apply the *same* mutation
  operators to full examples (host side included so they actually
  launch) and run each mutant under `compute-sanitizer --tool synccheck`,
  then compare three ways: agreement, static-only, dynamic-only.

Session #2 is, in miniature, the differential experiment this brief
describes. The laboratory should therefore be positioned as **the
generalization of session #2**, not as an unrelated new system — and
running those two sessions is the cheapest possible way to obtain the
empirical ground the laboratory needs anyway.

### 3.3 cuda-oxide already has a differential fuzzer — of something else

`crates/fuzzer` is rustlantis-based: it generates scalar custom-MIR
functions, runs the same case on CPU and GPU, and compares `u64` trace
hashes from `dump_var` sites. Statuses are `PASS` / `MISMATCH` /
`COMPILE_FAIL` / `UNSUPPORTED`.

The boundary is clean and worth stating precisely, because it is the
non-goal most likely to be violated by accident:

| | cuda-oxide `crates/fuzzer` | this laboratory |
|---|---|---|
| generates | scalar MIR programs (rustlantis) | SIMT kernels with declared convergence properties |
| threads | effectively one | 32 lanes, launch matrix |
| oracle | CPU execution of the same program | construction + static + GPU + sanitizer |
| target under test | the **codegen backend** | the **static analyzer's model** |
| a finding means | miscompilation | the analyzer's model diverges from evidence |

### 3.4 cuda-oxide already wraps Compute Sanitizer

`cargo oxide sanitize <example> --tool {memcheck,racecheck,initcheck,synccheck}`
exists (`crates/cargo-oxide/src/main.rs`, `SanitizerTool`), including
`--lineinfo` / `--device-debug` for source attribution, and it probes
`/usr/local/cuda/bin/compute-sanitizer` and friends when the binary is
not on `PATH`. The laboratory must shell out to *this*, not re-derive a
sanitizer invocation.

Arguments after `--` go to compute-sanitizer; a second `--` forwards to
the program. Verified in the clap declaration, not guessed.

---

## 4. Prior art, and the honest novelty claim

| Work | What it does | Why this project is not it |
|---|---|---|
| **GPUVerify** (OOPSLA'12) | Verifies barrier-divergence freedom and data-race freedom by reducing a kernel to a sequential two-thread program under predicated (SDV) semantics. | Predates CUDA 9 masked warp primitives entirely; has no notion of a participation mask. Also a verifier, not a differential harness. |
| **CUDAsmith** | Random CUDA program generation, differential testing across compilers, EMI-style variants. | Targets *compiler* correctness with random programs. No SIMT property oracle; a mismatch means miscompilation, not analyzer imprecision. |
| **NVIDIA cuFuzz** | Whole-program CUDA fuzzing with device-side coverage and sanitizer integration. | Targets applications, coverage-first. No static analyzer under test, no construction oracle. |
| **Compute Sanitizer `synccheck`** | Runtime detection of illegal `__syncthreads()` / `__syncwarp()` use, including the divergent-barrier case. | An evidence *source* for this project, not a competitor. Only sees the launch you ran. |
| **cuda-oxide `crates/fuzzer`** | rustlantis scalar MIR, CPU-vs-GPU trace hashes. | §3.3. |
| **reconverge conformance corpus** | Mutation operators over pinned examples, precision/recall published. | §3.1 — static only, never executed. |

The claim this project may make, and no more:

> An external, reproducible differential laboratory for a SIMT static
> analyzer, in which each case carries a construction-time semantic
> label, is executed on real hardware under a controlled launch matrix,
> is cross-checked against a vendor dynamic checker, and is minimized to
> an independently runnable reproducer.

The claim it may **not** make is that differential testing of GPU tools,
barrier-divergence analysis, or counterexample minimization is new. None
of them are.

## 5. The four evidence sources, instantiated

The brief's abstract model, bound to what actually exists here:

| Source | Concretely | Strength / what it cannot say |
|---|---|---|
| **A. Construction** | the generator template's declared class (`KNOWN_SAFE`, `KNOWN_UNSAFE`, `KNOWN_MASK_INVALID`, …) | strongest for *unsafety by construction*; must never be derived from what reconverge said |
| **B. Static** | `findings.v1` JSONL + `witness.v1` files + exit code + stderr, at a recorded tool version | says what the analyzer asserts, at which tier; a `warning` is not an assertion |
| **C. Dynamic** | `cargo oxide run` in a child process under `timeout`, plus values written back to the host | scoped to one input × one launch × one GPU; a clean run proves nothing universal |
| **D. Instrumentation** | `cargo oxide sanitize --tool synccheck`; later, device-buffer traces | synccheck sees the launch it ran; it is also *instrumentation*, so disagreement with the raw run is `InstrumentationConflict`, not an analyzer verdict |

The rule that keeps this honest, restated for this specific stack: for a
false-positive claim, **the finding must be `deny` or `confirmed`**
(§2.1) — a `warning`-tier finding under a documented limitation (§2.4)
is the tool working as specified, and belongs in `AnalyzerUnsupported`.

## 6. Scope decisions taken

1. **Target RC001 and RC002 only.** RC003 is syntactic (100% recall
   already), RC004 is a capacity check, RC005 is a contract check. None
   of the three has an interesting dynamic counterpart.
2. **Track upstream `main`, record the revision per case.** reconverge
   pins `a766fc26` for conformance; a laboratory that also pinned would
   inherit a stale surface. Record both revisions in every case.
3. **Use `cargo oxide sanitize`, never a hand-rolled compute-sanitizer
   command line** (§3.4).
4. **Consume only reconverge's published interfaces** — JSONL, SARIF,
   the artifact files in `<target>/reconverge/`. No linking against
   `reconverge-core`, which is behind a dialect-agnostic trait boundary
   its own CI gate (`scripts/check-isolation.sh`) protects.
5. **NVBit is out of scope until a question demands it.** Source-level
   device-buffer tracing first.

---

## 7. Classification, bound to this analyzer

The brief's enum, with the entry conditions this stack actually permits.
`tier` is reconverge's confidence; `oracle` is the generator's class.

| Classification | Entry condition |
|---|---|
| `AgreementSafe` | `oracle=KNOWN_SAFE` · no finding at deny/confirmed · GPU completed · synccheck clean |
| `AgreementBug` | `oracle=KNOWN_UNSAFE` · finding at confirmed · synccheck reported |
| `ConfirmedStaticBugDynamicObserved` | finding at confirmed · dynamic evidence matches the predicted lane split |
| `PotentialFalseNegative` | §31 protocol, **and** the construct is inside the surface of §2.3 **and** not excluded by a limitation of §2.4 |
| `PotentialFalsePositive` | finding at **deny or confirmed** (never warning) · safe by construction · clean across the whole generated finite matrix |
| `AnalyzerUnsupported` | the construct is named in §2.3 as outside v1, or §2.4 predicts the degradation observed |
| `ConstructionOracleConflict` | dynamic evidence contradicts the template's declared class — the *template* is suspect first, not the analyzer |
| `InstrumentationConflict` | raw run and sanitized run disagree |
| `DynamicInconclusive` | GPU completed but the property is not observable in the outputs collected |
| `NondeterministicObservation` | repeated identical runs disagree |
| `GpuTimeout` | watchdog fired — **records that the watchdog fired, nothing more** |

Two rules are worth writing into the code rather than the docs:

- A `warning`-tier finding can never produce `PotentialFalsePositive`.
- `GpuTimeout` is never by itself promoted to "deadlock proven". The
  probe script in this repo already enforces the wording (`watchdog-fired`).

## 8. What the first milestone actually builds

Given §3, the first milestone is smaller than the brief implies, because
two of its stages already have upstream equivalents to reuse:

| Stage | Build | Reuse |
|---|---|---|
| 0 | this document | — |
| 1 | one KNOWN_UNSAFE case end to end | reconverge's own `divergent_barrier` sample kernel, verbatim |
| 2 | the safe counterpart, and a classifier that separates them without consulting reconverge | — |
| 3 | ~10 templates over RC001/RC002 | operator names from `conformance/MUTATION.md` so the two corpora stay comparable |
| 4 | deterministic mutation | the five existing operators as the starting vocabulary |
| 5 | AST minimization | — |
| 6 | device-buffer tracing | — |

Deliberately **not** in the first milestone: coverage-guided generation,
NVBit, RC003/RC004/RC005, multi-GPU, cluster/grid barriers (they need
launch features whose availability must be capability-checked first).

---

## 9. Open empirical questions, and the probe that answers them

Four facts the design depends on cannot be established by reading code.
The probe (`probes/` in this repo, run on a rented A10G) answers them.

| # | Question | Why the design depends on it |
|---|---|---|
| Q1 | Is `compute-sanitizer` present, and does it carry `synccheck`? | it is evidence source D |
| Q2 | Does a cuda-oxide divergent `sync_threads()` **hang** or **complete** on sm_86? | if it completes, "GPU timeout" is not the dynamic signal for RC001 and the whole dynamic oracle must lean on synccheck |
| Q3 | Does `synccheck` flag it, with what wording and source attribution? | it determines whether a `PotentialFalseNegative` can ever be evidenced |
| Q4 | Is a mask naming fewer lanes than participate (`shrinkmask`, 0% static recall) observable at all — wrong value, or sanitizer report? | it decides whether that published static gap has a dynamic counterpart worth filing |

### 9.1 Measured environment

| | |
|---|---|
| GPU | NVIDIA A10G, compute capability **8.6**, driver 595.71.05 |
| CUDA | 13.2 (V13.2.51) |
| Compute Sanitizer | **2026.1.0.0**, tools: memcheck, racecheck, **synccheck** |
| rustc | 1.96.0-nightly (55e86c996 2026-04-02) |
| cuda-oxide | `50d07314` (2026-08-18), recorded on the controller |

**Q1: answered — yes.** `synccheck` is present and is one of three tools.

### 9.2 A harness bug worth keeping in the record

The first probe run reported `watchdog-fired` for **24 of 24** cases,
including `probe_safe_barrier`, which is safe by construction and should
finish in milliseconds. The cause was in the harness, not the GPU: the
watchdog wrapped `cargo oxide run`, which *compiles before it runs*, so
a 25-second watchdog was killing dependency compilation.

Two things are worth extracting:

- **Build time and execution time must be timed separately**, and only
  execution may carry a short watchdog. The probe now builds with a
  1800 s watchdog and executes with 20 s.
- **The wording discipline saved the data.** Because the runner recorded
  `watchdog-fired` rather than `deadlock` or `hang`, the bad run produced
  no false claim — it produced an implausible uniformity that was
  immediately readable as an infrastructure failure. This is the
  `InfrastructureFailure` class earning its place before a single
  generated case exists: had the harness written "GPU deadlock detected",
  the same run would have manufactured 24 fictitious bugs, one of them
  for a kernel that is provably fine.

### 9.3 Q2 — the divergent barrier does **not** hang on sm_86

All five probes completed in under a second. Every launch wrote every
element; nothing timed out.

| probe | oracle | block | raw run | synccheck |
|---|---|---:|---|---|
| `probe_safe_barrier` | KNOWN_SAFE | 32/64/128 | completed, 128/128 written | 0 errors |
| `probe_divergent_barrier` | KNOWN_UNSAFE (intra-warp) | 32/64/128 | **completed**, all written | **0 errors** |
| `probe_warp_divergent_barrier` | KNOWN_UNSAFE (whole-warp) | 64/128 | **completed**, all written | **0 errors** |
| `probe_mask_full` | KNOWN_MASK_VALID | 32/64/128 | completed, `0xffffffff` | 0 errors |
| `probe_mask_shrunk` | KNOWN_MASK_INVALID | 32/64/128 | completed, **`0xffffffff`** | **0 errors** |

`probe_divergent_barrier` is reconverge's own canonical sample kernel,
copied verbatim.

**This is not the compiler optimizing the barrier away.** The emitted PTX
keeps it inside the predicated branch:

```ptx
; probe_divergent_barrier
setp.ne.b64  %p8, %rd9, 0;
@%p8 bra     $L__BB0_2;      ; odd lanes jump over
bar.sync     0;              ; only the even lanes execute this

; probe_warp_divergent_barrier
setp.gt.u64  %p8, %rd2, 31;
@%p8 bra     $L__BB0_2;      ; warp 1 jumps over
bar.sync     0;
```

The divergence reaches the hardware and the hardware completes anyway.
For the whole-warp case the mechanism is the ordinary one: `bar.sync`
waits for non-exited threads, and the warp that skips the barrier runs
to the end of the kernel and exits, which satisfies it.

**Consequences for the design, and they are large:**

1. **A watchdog timeout is not the dynamic signal for RC001** on this
   architecture. A laboratory built on "unsafe kernels hang" would have
   found nothing, forever, and concluded the analyzer was over-reporting.
2. `AgreementBug` for RC001 cannot require dynamic failure. The correct
   classification for these cases is agreement between the construction
   oracle and the static finding, with the dynamic run recorded as
   *observed-clean-under-this-launch* — which proves nothing universal
   and must not be allowed to argue against the static finding.
3. This is exactly the datum reconverge's `docs/hardware/session-1.md`
   was written to collect. Its witness verdict says "undefined behavior,
   usually a permanent hang"; on sm_86 the observed behaviour for its own
   sample kernel is *completion*. The wording is defensible (UB is UB),
   but "usually" is now measured rather than assumed, and the honest
   phrasing is closer to "undefined behaviour; may hang, and on Ampere
   this shape did not".

### 9.4 Q3 — `synccheck` is silent on all five

Compute Sanitizer 2026.1.0.0 reported `ERROR SUMMARY: 0 errors` for every
probe, including both divergent-barrier shapes and the invalid mask.

So on this stack, `synccheck` is **not** a usable oracle for the RC001
shapes reconverge targets. Evidence source D is weaker than the brief
assumes, and the laboratory cannot lean on it for false-negative
protocols. This needs re-probing on other architectures and other block
geometries before it is stated as a general fact — recorded here as
measured on one GPU, one driver, one toolkit.

### 9.5 Q4 — the invalid mask is silently wrong, and only construction knows

`probe_mask_shrunk` calls `ballot_sync(0x0000_ffff, true)` with all 32
lanes present. The mask **is** honoured in the emitted PTX:

```ptx
vote.sync.ballot.b32  %r1, %p9, 65535;   ; 65535 = 0x0000ffff
```

so cuda-oxide is lowering it faithfully — an earlier suspicion that the
mask operand was being dropped was wrong, and the PTX is what settled it.
Every lane read back `0xffffffff`, which is **byte-identical to what the
valid-mask probe returns**.

That is the most useful single result of this probe:

- the raw output cannot distinguish the valid case from the invalid one;
- `synccheck` reports nothing;
- reconverge's own corpus records this class (`shrinkmask`) at **0%**
  static recall, by design, in v1.

Three independent mechanisms are silent, and the only thing that knows
the program is wrong is **the construction oracle** — the generator's
record that it built a mask naming 16 lanes for a call all 32 execute.
This is the concrete justification for evidence source A being
first-class rather than a convenience, and it is the strongest argument
in this document for building the laboratory at all.

### 9.6 Throughput, measured

| step | time |
|---|---|
| first build (cold backend) | 75 s |
| subsequent probe builds | **4 s** each |
| kernel execution | < 1 s |
| execution under synccheck | ~1 s |

A 4-second per-case build cost sets the realistic scale of the first
experiments: ~900 cases/hour of pure build on one box, before any
generation or analysis. The brief's ladder (10 → 100 → 1 000 → 10 000)
is achievable, but 10 000 cases is a multi-hour GPU booking, not an
afternoon.

---

## 10. What the probe changes about the plan

Three revisions to the brief, each forced by a measurement above.

1. **Demote the dynamic-failure oracle; promote the construction
   oracle.** The brief's canonical example is "GPU times out, sanitizer
   reports, reconverge is silent → potential false negative". On sm_86
   the GPU does not time out and the sanitizer does not report, for the
   analyzer's own flagship bug class. The laboratory's discriminating
   power therefore comes from the semantic label plus *value*
   comparison against a reference model, not from crash observation.

2. **Add a reference-model comparison to every collective case.** §9.5
   shows the invalid mask is invisible unless something knows what the
   result *should* have been. Every `KNOWN_MASK_*` template must ship an
   expected-value computation so the runner compares values rather than
   exit codes.

3. **Reorder the milestone.** Stage 1 in the brief is "one KNOWN_UNSAFE
   divergent barrier end to end". That case, on this hardware, produces
   no dynamic evidence at all — a poor first vertical slice. Stage 1
   should instead be the **mask pair** (`probe_mask_full` vs
   `probe_mask_shrunk`), because it exercises every part of the pipeline
   *and* has a real, checkable discrepancy at the end of it.

## 11. Honest status of this document

- §2, §3, §5–§8 are read from source at the revisions in the header.
- §4 draws on published descriptions of GPUVerify, CUDAsmith, cuFuzz and
  Compute Sanitizer; the comparisons are structural, not benchmarked.
- §9 is measured on **one** GPU (A10G, sm_86), **one** driver (595.71.05),
  **one** toolkit (CUDA 13.2), **one** sanitizer (2026.1.0.0). Nothing in
  it should be restated as architecture-independent. sm_75, sm_90 and
  sm_100 are unmeasured, and the whole-warp barrier result in particular
  deserves re-running where `bar.sync` participation rules differ.
- No claim here has been filed upstream. The session-1 wording
  observation in §9.3 is a candidate, and belongs to reconverge's own
  hardware-session process, not to this repository.
