# Stage 5 — making findings survive

Stage 4 produced five findings and, along the way, five successive versions of
one of them. Four were wrong. That is the problem this stage is for: a finding
that lives only in a document decays quietly, and a claim nobody re-checks is
indistinguishable from a claim that has stopped being true.

Three tools, all on a laptop, no GPU:

| tool | what it does |
|---|---|
| `simt-diff minimize` | shrinks a case while preserving what makes it interesting |
| `simt-diff package` | writes a standalone reproducer a stranger can run |
| `simt-diff corpus add` / `regress` | records a finding with a machine-checkable expiry |

## Minimizing

Delta debugging is normally delicate work on text: most candidates do not parse,
and a "smaller" version may not mean the same thing. Neither problem exists here.
Candidates are `ir::Kernel` values, so they always render to a valid program, and
their oracle is **recomputed** rather than assumed to survive.

That makes the preserved property the only real decision, and the default is
deliberately strict:

> **the construction oracle *and* the analyzer's signature**

Signature alone would be a trap. These findings are observations — "RC001 at
warning tier with no witness, on a kernel construction says is unsafe" — and a
reduction that quietly turned the kernel *safe* would preserve the analyzer's
answer while destroying the finding. `--signature-only` exists for when that is
genuinely what you want, and says so in its own help text.

Run on the `A, B, A` case, the hardest of the five findings:

```
Size 16 -> 13 nodes, in 3 accepted reduction(s) over 41 analyzer run(s).
```

```rust
if i.get() == 0     { thread::sync_threads(); }
if i.get() % 2 == 0 { thread::sync_threads(); }
if i.get() == 0     { thread::sync_threads(); }
```

The report also lists every **rejected** reduction with what it turned the case
into, because that list is the shape of the boundary — each line is a program that
is *almost* this finding and is not. One of those rejected lines turned out to be
new evidence: `A, A, B` came back with two `confirmed` findings and two witness
artifacts, corroborating the prefix rule from a mechanism that was not looking for
it.

## Packaging

A reproducer directory states the argument, not just the code: the kernel, what
construction knows and how it was computed, what was expected **and on whose
authority** (quoted / extrapolated / measured), what actually happened, the exact
commands, and what the case does *not* show.

`verify.sh` re-runs the analyzer and exits nonzero if the observation has moved. It
parses the analyzer's own JSON with python3 rather than grepping it, because a
reproducer that silently passes when the output format changes is worse than one
that fails to run — and it exits 2, not 0, when it cannot run at all. It needs no
part of this repository:

```
$ ./verify.sh
expected: rc001/warning,rc001/warning|0w
observed: rc001/warning,rc001/warning|0w
OK: the observation is unchanged
```

## The regression corpus

An entry stores the **recipe** — a seed template and the operators applied to it —
not the program. Rebuilding from the recipe and comparing the kernel hash catches
the second kind of decay, where the generator changes and a case silently becomes
a different case. The two failures are reported separately because they mean
opposite things:

- **ANALYZER DRIFT** — same program, different answer. News about reconverge, and
  the reason the entry exists.
- **GENERATOR DRIFT** — the recipe now builds something else. News about this
  repository.

Getting that distinction working immediately caught a bug in the checker itself:
the rebuild dropped the mutation lineage, which is part of the generated kernel's
doc comment, so *every* entry looked like generator drift. A drift detector that
cries wolf on every entry is worse than none.

### The nine entries

Every finding is paired with the case that bounds it. A finding without its
contrast is an anecdote.

| entry | signature | why it is here |
|---|---|---|
| `gate-baseline-confirmable-barrier` | `rc001/confirmed\|1w` | the control: this is what a working gate looks like |
| `ci-gate-lost-to-a-barrier-above` | `rc001/warning,rc001/warning\|0w` | the same barrier, un-gated by one added above it |
| `lane-env-barrier-below-is-harmless` | `rc001/confirmed,rc001/warning\|1w` | added below instead: keeps the claim ordered, not function-wide |
| `guard-inside-loop-not-promoted` | `rc001/warning\|0w` | a divergent guard inside a loop is never promoted |
| `loop-inside-guard-is-promoted` | `rc001/confirmed\|1w` | the same constructs nested the other way |
| `correct-partial-warp-mask-not-gated` | `rc002/warning\|0w` | the valid idiom is reported and never gated |
| `truncating-cast-is-evaluated` | `rc001/confirmed\|1w` | casts are evaluated, contrary to the README's stated reason |
| `unreachable-barrier-still-reported` | `rc001/warning\|0w` | a barrier no thread reaches is still reported |
| `witness-reaches-only-a-prefix` | `rc001/confirmed,rc001/warning,rc001/warning\|1w` | the case that refuted two versions of the ordering rule |

```
9 entry(ies): 9 unchanged, 0 moved
```

Each rebuilds from its recipe to a byte-identical kernel and reproduces its
recorded signature against reconverge 0.1.6.

## What this stage does not do

- **Nothing is filed upstream.** The packaged reproducer and the corpus are what
  a report would rest on; deciding to open an issue against `vyncint/reconverge`
  is not a decision this stage makes.
- **The corpus pins one analyzer version and one launch.** Every entry says which,
  in its own file. An entry that moves after a version bump is doing its job.
- **`regress` does not run on a GPU.** Every entry is a static observation. The
  dynamic half of the laboratory still has no regression coverage, because it has
  no launch matrix yet.
- **Minimization is greedy, not `ddmin`.** Each candidate costs an analyzer run of
  a few seconds and these kernels are small; the difference in result is nil, and
  the cost of pretending otherwise is not.
