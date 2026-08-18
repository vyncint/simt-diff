# Stage 1 — the mask pair, end to end

Status: **complete.** One vertical slice runs from generation to a classified
verdict, with real reconverge and a real A10G.

Stage 1 was reordered from the brief on purpose: the brief's first slice is a
divergent barrier, and the baseline measured that producing *no* dynamic
evidence on sm_86 (§9.3/§9.4). The mask pair exercises the same pipeline and
ends in a real, checkable discrepancy.

## The slice

```
templates::mask_{full,shrunk}_convergent
  -> generate  : two crates per case, one analyzable anywhere, one for the GPU
  -> analyze   : real `cargo reconverge check --message-format json --strict`
  -> [GPU host]: cargo oxide build + run, then compute-sanitizer synccheck
  -> ingest    : parse BLOCK=/VALUES= and the sanitizer summary into records
  -> compare   : classify from the evidence, and state what is not claimed
```

## Result

| | valid mask | invalid mask |
|---|---|---|
| oracle | `KNOWN_MASK_VALID` | `KNOWN_MASK_INVALID` |
| reconverge 0.1.6 | no finding | **no finding** |
| GPU (A10G, sm_86) | completed | completed |
| synccheck | 0 errors | **0 errors** |
| observed value | `0xffffffff` (correct) | **`0xffffffff`** (reference: `0x0000ffff`) |
| classification | `AgreementSafe` | `AnalyzerUnsupported` |

The two kernels are identical apart from the mask literal, and a test asserts
that (`the_mask_pair_differs_only_in_the_mask_literal`), so the difference in
evidence is attributable to the mask alone.

## What this case teaches

**The value comparison is the only channel that sees it.** Static: silent.
Sanitizer: silent. Exit code: success. Output byte pattern: identical to the
valid case. Only the template's reference model — construction knowledge —
distinguishes them.

**And it is still not a bug report.** The first version of the classifier
called this `PotentialFalseNegative`. That was wrong, and the fix is the most
important thing in this stage: reconverge's own `conformance/MUTATION.md`
publishes the `shrinkmask` class at **0% recall in v1**, with a stated
reason. Filing it would be filing a bug against a documented limitation.

`Template::documented_limitation` now carries that quote, and its presence
routes the case to `AnalyzerUnsupported` while *keeping* the dynamic
evidence, because the evidence says something the static gap does not: when
this class is missed, hardware does not complain either. Two golden tests pin
the distinction — with the limitation declared it is `AnalyzerUnsupported`,
without it the same evidence is `PotentialFalseNegative`.

## Positive control

A laboratory that reports "no finding" must prove the analyzer can see its
generated crates at all. `barrier_divergent_intra_warp` returns
`RC001/confirmed`, exit 1, and one `witness.v1` artifact — so silence on the
mask pair is a real absence.

| template | reconverge verdict |
|---|---|
| `mask_full_convergent` | no findings, exit 0 |
| `mask_shrunk_convergent` | no findings, exit 0 |
| `barrier_uniform` | no findings, exit 0 |
| `barrier_divergent_intra_warp` | **RC001 confirmed**, exit 1, 1 witness |

## Two things the implementation learned the hard way

- **The kernel crate and the runner crate need different dependency specs.**
  Wiring both to a path dep made `cargo metadata` fail on the controller, so
  every case classified `AnalyzerError`. The classifier was right to refuse a
  verdict; the emitter was wrong. A regression test now asserts the kernel
  crate stays portable.
- **`#[launch_contract]` changes the host API.** A contracted kernel launches
  through `prepare_<kernel>(LaunchConfig1D)` and takes `&PreparedLaunch`, not
  a bare `LaunchConfig`. Read out of upstream's `coop_groups_demo` after the
  generated runner failed to compile — the brief's "do not invent launch
  syntax" rule, earning itself twice in one afternoon.

## Not done yet

Mutation engine, minimizer, corpus, reports, issue drafts, device-buffer
tracing, and the launch matrix beyond a single block size. `simt-diff run`
does not exist: execution happens on the GPU host and enters through
`ingest`, which is honest about provenance but not yet automated.
