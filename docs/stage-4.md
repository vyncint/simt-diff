# Stage 4 — the mutation engine

Stage 3 wrote fourteen kernels by hand, predicted what reconverge would say
about each, and was right fourteen times. That is a real result and it is also
a ceiling: a hand-written corpus can only contain constructs somebody thought
of. The engine in `crates/simt-diff/src/{ir,interpret,model,mutate}.rs` exists
to generate the ones nobody did.

Everything here runs on a laptop. No GPU was used at any point in Stage 4.

## The problem with mutating a labelled case

The brief's §11 is blunt about it: a mutation invalidates the label. Take the
divergent-barrier template, remove its guard, and the case is still named
`barrier_divergent_intra_warp` while the program is now perfectly safe. Ship
that to an analyzer and any silence looks like a false negative. reconverge's
own mutation corpus keeps a row for exactly this class (`delbar`, 0/67 by
design), and the reason it can is that a human checked each one.

So the engine does not mutate *labelled cases*. It mutates programs, and then
recomputes the label:

| stage | what it produces | how |
|---|---|---|
| `ir.rs` | a closed representation of a probe kernel | small enough to execute exactly |
| `interpret.rs` | the oracle, the reference model, per-site facts | runs every thread of the launch |
| `model.rs` | what reconverge should say, and why | rules keyed to its documentation |
| `mutate.rs` | the corpus | operators that cross documented boundaries |

`interpret.rs` executes the mutant over all 32 threads and derives the oracle
from what happened. A guard that no longer diverges yields `KNOWN_SAFE`, with a
sentence saying how many threads reached which barrier. Nothing is inherited.

Two modelling decisions in the interpreter are conservative on purpose, and
both are recorded in the module documentation rather than left implicit:

- **Complementary guards are still unsafe.** `% 2 == 0` in one branch and
  `% 2 != 0` in another gives every thread exactly one barrier arrival, and
  hardware arrival counting might well let it through. The CUDA programming
  model calls a barrier under divergent control undefined regardless, and a
  laboratory does not get to assume hardware charity.
- **No reference model where execution is undefined.** If a barrier is
  divergent, or a mask names an absent lane, the program has no defined result
  and predicting values would be inventing evidence — the §9.2 mistake.

A third decision was a bug, found by reading the engine's own output: the
barrier inside a helper was keyed once globally, so *two* complementary call
sites looked uniform while two inlined barriers looked divergent. Same program,
two labels. The call site is now the site, because the call site is where the
thread's arrival happens.

## Three kinds of prediction, and why the difference matters

Stage 3 declared one prediction per template by hand. At 144 cases that is not
possible, and hand-declaring a mutant's prediction would repeat the sin of
hand-declaring its oracle. So the rules are written once, as a function of a
kernel's static features — and each rule carries its provenance:

- **`quoted`** — the documentation states this behaviour for this construct. A
  violation is a statement about reconverge.
- **`extrapolated`** — the documentation states a *reason*, and I inferred what
  it implies here. A violation is first a statement about this model. Reporting
  one as an analyzer bug would be dishonest.
- **`measured`** — undocumented, but observed by this laboratory, with the case
  that established it named in the rule. A violation of one of these is a
  *regression*, which is a third and more actionable claim.

The `measured` tier did not exist when Stage 4 started. It was added because
the first sweep produced findings that were neither "reconverge is wrong" nor
"my inference was wrong", but "nobody wrote this down and here is what it does".

The model consults executed facts in exactly one place, where reconverge's own
mechanism does: the witness interpreter replays 32 lanes, so a guard it can
evaluate and finds uniform gives it no divergent lane pair to promote. Modelling
that is modelling a documented mechanism, not peeking at the answer.

## What licenses trusting the engine

`tests/ir_seeds.rs`. Every one of the fourteen Stage 3 templates has an IR seed,
and each seed must

1. render **the same program**, byte for byte, as the hand-written template that
   was measured;
2. compute **the same oracle** the template declared; and
3. derive **the same prediction**, so the model reproduces all fourteen measured
   rows.

If a rule change breaks one of those, it broke a measurement, and the test says
so in those words. The engine's verdicts on kernels nobody has run rest on
reproducing the ones somebody did.

### The one seed that did not match

`barrier_guarded_by_warp_id` declares `KNOWN_UNSAFE`, justified as "warp_id() is
uniform within a warp but differs across warps, so whole warps skip the block
barrier". That needs more than one warp. Every Stage 3 row was measured at
`block=32`, where the guard is true for every thread and *nothing diverges at
all*. The interpreter says `KNOWN_SAFE` at 32 threads and `KNOWN_UNSAFE` at 64,
and it is right both times.

The analyzer's behaviour is unaffected — a lane-environment guard is a
divergence source it cannot evaluate, so RC001 at warning tier is correct at
either launch, which is what was measured and why the Stage 3 row held. What
changed is that the construction label is now computed per launch instead of
asserted once. The hand-written template keeps its declaration; the discrepancy
is pinned by a test named after it rather than smoothed away.

## The operators

Each one moves a kernel across a boundary the documentation draws, rather than
perturbing it randomly.

| operator | what it crosses |
|---|---|
| `invert_guard`, `negate_cmp` | the same condition, two spellings — a syntactic-robustness check |
| `shift_bound`, `retarget_modulus` | which lanes diverge, keeping the guard evaluable |
| `truncate_operand` | onto a truncating cast, named in the README as missing machinery |
| `to_warp_id`, `to_lanemask` | same divergence, onto a source documented as unevaluable |
| `conjoin_lane_env` | one evaluable predicate and one unevaluable one in the same guard |
| `hoist_into_helper`, `deepen_helper` | one call, then two, between the divergence and the barrier |
| `nest_guard`, `wrap_in_loop`, `uniform_bound` | control shapes: nesting, divergent trip counts, uniform trip counts |
| `drop_guard` | removes the bug entirely — the `delbar` control |
| `complementary_guard` | every thread arrives exactly once, at one of two points |
| `mask_widen/shrink/single_lane` | mask arithmetic against a fixed participant set |
| `mask_to_named_const/active_mask/wrapper` | mask provenance: literal, const, computed, implicit |
| `mask_match_participants` | **the correct guarded partial-warp idiom, spelled as a literal** |

Identity is the SHA-256 of the rendered program, so two operator chains that
reach the same text are one case. Sampling uses a named seed and reports what it
dropped: a sweep that silently truncates reads as "covered everything" when it
did not.

The prediction is deliberately *not* written into the generated kernel, only
into `generator.json`. Case identity is the hash of the source, so a rule change
would otherwise rename every case in the corpus — and a regression corpus cannot
have identities that move like that.

## What the sweep found

Two sweeps, both on a laptop, both against reconverge 0.1.6 at `block=32`: the
whole depth-1 corpus, and a depth-2 corpus restricted to one seed in order to
minimize a boundary the first sweep exposed.

The headline is a negative result, and it is the one that matters most:

> **No false positive at a gating tier, in any case.** Across the corpus,
> including cases built specifically to force one, reconverge never gated on a
> program that construction says is valid. Its README calls zero false positives
> at default confidence "a requirement, not a goal"; this is independent evidence
> for that claim, from kernels the project did not write.

Then five findings. Every one of them is a place where reconverge's behaviour
and reconverge's documentation are not the same thing — in three cases because
the tool is *better* than the documentation says.

### 1. RC002 does compare the mask against the lanes present

`conformance/MUTATION.md` says v1 "checks convergence; it does not do mask
arithmetic against launch shapes", and the `shrinkmask` row is published at 0%
recall on that basis. Read literally, that predicts a gating RC002 for **any**
collective at a call site that cannot be proven convergent — including the
correct guarded partial-warp idiom:

```rust
if i.get() % 2 == 0 {
    b = warp::ballot_sync(0x5555_5555, true);   // names exactly the lanes present
}
```

That program is valid by construction, so a gating finding would be a false
positive. It was reported **RC002 at warning tier, and never gated.**

Sorting every RC002 finding in the depth-1 corpus by tier gives a clean split:
all fourteen promoted to `confirmed` were calls where **some lane the mask names
is absent**; none of the twelve held at warning tier were, except where an
independent documented rule (lane-environment guard, non-literal mask) already
explains the tier. So the promotion criterion is mask-versus-participants, which
the documentation says v1 does not do. The tool is more capable than its own
description, and the benefit is precision.

### 2. Only one divergence source per function gets a witness, and it is the first one

This is the finding with a consequence, and it took **four** attempts to state
correctly. Three of those attempts were wrong, and each was refuted by a case
built to demonstrate it. That sequence is the most useful thing in this document,
so it is written out rather than tidied away.

**The control.** Two barriers, both under guards the analyzer can evaluate:

```rust
if i.get() % 2 == 0 { thread::sync_threads(); }        // RC001/confirmed, 1 witness
if !(i.get() % 2 == 0) { thread::sync_threads(); }     // RC001/warning
```

Only the first is promoted, though both are equally divergent and equally
evaluable.

**Attempt 1: "an unevaluable guard blocks the whole function."** Replacing the
*first* guard with a lane-environment read drops both findings to warning tier
with no witness — including the second, which is unchanged and independently
confirmable. Three operators reach that shape, and so does putting the first
guard inside a loop. So: one unevaluable source anywhere suppresses everything.

To make it reportable, `add_lane_env_sibling` was written to generate the
cleanest demonstration — original guard untouched, one lane-environment barrier
added beside it:

```rust
if i.get() % 2 == 0 { thread::sync_threads(); }        // RC001/confirmed, 1 witness
if warp::warp_id() == 0 { thread::sync_threads(); }    // RC001/warning
```

**The reproducer refuted the claim.** The first barrier is still confirmed.
Position matters, not mere presence.

**Attempt 2: "only the first finding is ever promoted."** Dead within one query:
three cases in the corpus carry **two** witnesses and two `confirmed` findings —
all of them shapes where the two barriers sit under *identical* guards.

**Attempt 3: "an unevaluable source *earlier* in the function suppresses later
promotion."** This fitted every multi-site case in both sweeps, so it was
adopted. Then a **held-out** corpus — 60 depth-2 mutants of a seed family the
rules had never been fitted to — produced one violation:

```rust
if i.get() % 2 == 0  { b = warp::ballot_sync(0x0000_0001, true); }   // fine: lane 0 is present
if !(i.get() % 2 == 0) { b = warp::ballot_sync(0x0000_0001, true); } // names lane 0, which is absent
```

Predicted a gating RC002 on the second call. Measured: warning on both. No
unevaluable guard appears anywhere, so attempt 3 could not explain it.

**Attempt 4: "the witness pass attempts the first divergence source in program
order and promotes exactly the findings that share it."** This explained every
multi-site case in all three corpora, including the held-out one, so it was
adopted — and then the experiment named at the end of this section broke it.
`A, B, A`, built by cloning the first guard to the end past a different one:

```rust
if i.get() == 0     { thread::sync_threads(); }   // RC001/confirmed, 1 witness
if i.get() % 2 == 0 { thread::sync_threads(); }   // RC001/warning
if i.get() == 0     { thread::sync_threads(); }   // RC001/warning  <-- same source as the first
```

The third site's source *is* the witnessed one and it is still not promoted.

**Attempt 5, which currently stands.** Promotion reaches an unbroken **prefix**
of the divergence sources in program order — call sites skipped — and stops at
the first source that differs from the first one.

| shape | promoted |
|---|---|
| `A, B` | the first only |
| `A, A` | both, two witness artifacts |
| `A, A, B` | the first two, two witnesses |
| `A, B, A` | the first only |
| `A, !A, !A` | the first only |
| lane-environment source first | nothing at all |
| divergent guard inside a loop, first | nothing at all |
| call site first, evaluable second | the second (call sites are not attempts) |

The CI consequence is narrower than attempt 1 claimed and still real: **a
`warp_id()`-guarded barrier added above an existing confirmable one silently
takes it out of the gate.** Nothing in the output says so. Added *below*, it
changes nothing — both directions are in the regression corpus, as
`ci-gate-lost-to-a-barrier-above` and `lane-env-barrier-below-is-harmless`.

The `A, A, B` row is independent corroboration that arrived by accident: the
minimizer produced it as a *rejected* candidate while shrinking `A, B, A`, and it
came back with two confirmed findings and two witnesses. It was generated by a
different mechanism than the sweep and agrees.

The *mechanism* is still unexplained — why the pass commits to a prefix rather
than trying each source — and the rule that encodes this says so in its own text.
The next experiment that would break it: a case where the differing source sits
in a branch no thread of the launch reaches, which tests whether "in program
order" means lexical or dynamic.

### 3. A divergent guard inside a loop is never witness-promoted; a loop inside the guard is

Minimized in the depth-2 sweep to a pair that differs only in nesting order:

```rust
while n < (2) { if i.get() % 2 == 0 { thread::sync_threads(); } }   // RC001/warning, 0 witnesses
if i.get() % 2 == 0 { while n < (2) { thread::sync_threads(); } }   // RC001/confirmed, 1 witness
```

Both programs are equally undefined. The loop's trip count is irrelevant — a
uniform bound (`n < 2`) and a thread-derived one (`n < i.get() % 4`) behave
identically — and a barrier directly inside a divergent loop, with no guard, is
promoted normally. It is specifically a *divergent `if` nested inside a loop*
that the replay does not follow.

`## Limitations` in reconverge's README does not mention loops at all.

### 4. The witness interpreter does evaluate truncating casts

The README explains the lane-environment gap as needing "width-typed evaluation
of integer `!` and truncating casts". Extrapolating from that, this laboratory
predicted that a guard needing a truncating cast would not be promoted:

```rust
if (i.get() as u8) as u32 % 2 == 0 { thread::sync_threads(); }   // RC001/confirmed, 1 witness
```

It was promoted, with a witness. Casts on the thread index are handled; whatever
is missing for the lane-environment registers, it is not that. The prediction was
labelled `extrapolated` before the run, so the violation landed where it belongs
— on the model — and the rule now cites the measurement instead of the README.

### 5. A barrier no thread can reach is still reported

```rust
if !(i.get() % 2 == 0) {
    if i.get() % 4 == 0 { thread::sync_threads(); }   // no lane satisfies both
}
```

RC001 at warning tier, no witness. The syntactic recognizer speaks and the replay
correctly declines to promote — defensible, since a launch contract is a
declaration and not a proof, and undocumented. It never gates, so by
reconverge's own confidence ladder it is not a false positive.

## What this does not establish

- One analyzer version, one pin, one launch shape (`grid=1, block=32`), one seed.
- **No GPU ran in Stage 4.** Every finding here is about the analyzer's static
  behaviour. The dynamic half of the laboratory was not exercised, so nothing
  here says what hardware does with any of these kernels.
- The corpus is one and two mutation steps from fourteen seeds. Depth 3 and
  beyond, other block sizes, multi-warp launches, and shared memory are all
  unexplored.
- Findings 2 and 3 are characterizations of *undocumented* behaviour. Neither is
  filed upstream yet; a report needs the minimal reproducer packaged and the
  behaviour confirmed against a second analyzer version, which is Stage 5 work
  (§21 minimizer, §22 reproducer packaging, §36 issue drafting).
- Nothing here is a claim about hardware, about other analyzers, or about
  constructs the IR cannot express — no shared memory, no atomics, no `asm!`, no
  multi-dimensional launches, one collective family.

## The sweeps, in numbers

Three corpora, all against reconverge 0.1.6 at `block=32`, all on a laptop.
Regenerate any table with
`scripts/conformance-summary.py <cases-dir>/conformance.json`, so these numbers
are read off the artifacts rather than remembered.

### The depth-1 corpus

### 147 cases, grouped by the rule that predicted them

| rule | provenance | held | violated | example violation |
|---|---|---:|---:|---|
| `barrier_under_evaluable_divergent_guard` | quoted | 24 | 0 |  |
| `barrier_under_guard_inside_loop` | measured | 1 | 0 |  |
| `barrier_under_lane_environment_guard` | quoted | 17 | 0 |  |
| `barrier_under_mixed_guard` | quoted | 15 | 0 |  |
| `barrier_under_truncating_cast_guard` | measured | 4 | 0 |  |
| `barrier_uniform_control` | quoted | 4 | 0 |  |
| `barrier_via_call_site` | quoted | 20 | 0 |  |
| `collective_at_convergent_call_site` | quoted | 5 | 0 |  |
| `collective_naming_an_absent_lane` | measured | 14 | 0 |  |
| `collective_under_guard_inside_loop` | measured | 1 | 0 |  |
| `collective_under_lane_environment_guard` | quoted | 3 | 0 |  |
| `collective_via_unmasked_wrapper` | quoted | 15 | 0 |  |
| `collective_with_every_named_lane_present` | measured | 2 | 0 |  |
| `collective_with_unevaluable_mask` | quoted | 6 | 0 |  |
| `construct_unreachable_at_this_launch` | measured | 13 | 0 |  |
| `outside_the_witnessed_prefix` | measured | 3 | 0 |  |

### classifications

| classification | cases |
|---|---:|
| AnalyzerUnsupported | 71 |
| AgreementBug | 42 |
| AgreementSafe | 34 |

**147 cases: 147 predictions held, 0 violated — 0 of those about the analyzer (a quoted or measured rule), 0 about this model (an extrapolated one). 0 case(s) classified as needing a human.**

### The held-out corpus — 60 depth-2 mutants of a seed family the rules were not fitted to

|---|---:|
| AgreementBug | 28 |
| AnalyzerUnsupported | 24 |
| AgreementSafe | 8 |

**60 cases: 60 predictions held, 0 violated — 0 of those about the analyzer (a quoted or measured rule), 0 about this model (an extrapolated one). 0 case(s) classified as needing a human.**

### The targeted corpus — the `clone_guard_to_end` family, which produced the `A, B, A` refutation

|---|---:|
| AgreementBug | 31 |
| AnalyzerUnsupported | 7 |
| AgreementSafe | 1 |

**39 cases: 39 predictions held, 0 violated — 0 of those about the analyzer (a quoted or measured rule), 0 about this model (an extrapolated one). 0 case(s) classified as needing a human.**

### What "all held" is and is not

The rules were **fitted to these cases.** Six of the sixteen were written from
observations in these very sweeps, so a clean sheet is a consistency check, not a
prediction — a model that failed here would simply be broken.

Two of the corpora did carry information, and both did it by *refuting* something:

- the 60-case held-out corpus killed the third version of the ordering rule;
- the 39-case targeted corpus killed the fourth.

Both are now part of the fit, so neither can validate what replaced it. Doing that
needs a corpus none of the rules has seen — depth 3, another seed family, or
another block size — and the single experiment most likely to break the current
version is named at the end of finding 2.
