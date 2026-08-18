# Contributing

Thanks for helping build simt-diff. [`docs/research-baseline.md`](docs/research-baseline.md)
explains what this laboratory is for and what it deliberately is not; this page
is the practical summary of how to work on it.

## Dev setup

1. Install [rustup](https://rustup.rs) and
   [`just`](https://github.com/casey/just) (plus
   [`cargo-deny`](https://github.com/EmbarkStudios/cargo-deny) for the full
   local CI run).
2. `just setup` — materializes the pinned nightly from `rust-toolchain.toml`
   and wires the repo-local git hooks.
3. `just ci` runs everything CI gates on. Keep it green before every push —
   never push red.

The analyzer under test is a separate build: point `SIMT_DIFF_RECONVERGE` at a
`cargo-reconverge` binary built from the same pinned nightly. Nothing in `just
ci` needs it. Nothing in this repository needs a GPU.

## The rules that make a claim worth reading

This project's whole output is claims about someone else's tool, so the bar is
the epistemics, not the code.

- **An oracle is computed, never inherited.** A generated kernel's semantic
  label comes from executing every thread of its launch
  ([`interpret.rs`](crates/simt-diff/src/interpret.rs)). A mutation that turns
  an unsafe kernel safe must produce a case labelled *safe*. Never carry a label
  across a transformation.
- **Every prediction states its provenance.** `quoted` (the documentation says
  so), `extrapolated` (inferred from a documented reason), or `measured` (this
  laboratory observed it, and the rule names the case). A violated
  `extrapolated` rule is a finding about *us*; a violated `quoted` or `measured`
  one is a finding about the analyzer. A new rule without a provenance will not
  be merged.
- **Only a gating-tier finding can be a false positive.** A `warning` is not an
  assertion. Reporting one as a false positive is reporting the tool for working
  as specified.
- **A clean GPU run never argues against a static finding.** A divergent
  `sync_threads()` completes on sm_86 and `synccheck` says nothing
  (`docs/research-baseline.md` §9.3–§9.4). Completion is not evidence of safety.
- **No silent caps.** If a sweep samples, truncates, or skips, it says so on
  stderr with the number it dropped.

## Testing policy

- [`tests/ir_seeds.rs`](crates/simt-diff/tests/ir_seeds.rs) holds the generator
  to the fourteen rows measured in Stage 3: each seed must render the same
  program byte for byte, compute the same oracle, and derive the same
  prediction. **If a change breaks one of those, it broke a measurement** — say
  so explicitly in the PR, or fix the change.
- A model rule that came from measurement ships with a test that pins both
  directions, not just the one that fired.
- A finding ships with the case that *bounds* it. A finding without its contrast
  is an anecdote.

## The regression corpus

[`corpus/`](corpus/) records findings with a machine-checkable expiry. Each
entry stores a recipe — a seed template plus mutation operators — and the hash
of the kernel it produced.

- `./scripts/check-corpus.sh` (CI) rebuilds every entry and compares hashes:
  **generator drift** is this repository's failure.
- `just regress` (local, needs the analyzer) re-runs each entry: **analyzer
  drift** is news about reconverge, and the reason the entry exists.
- Re-recording an entry is a decision. Say in the PR why the old observation
  stopped being the right one.

## Commit conventions

- [Conventional Commits](https://www.conventionalcommits.org), imperative
  subject ≤ 72 chars. Scopes: `ir:` `interpret:` `model:` `mutate:` `minimize:`
  `package:` `corpus:` `classify:` `cli:` `ci:` `docs:` `repo:`.
- **DCO**: sign off every commit (`git commit -s`); the `Signed-off-by:`
  trailer must match the author identity.
- **AI assistance is welcome; AI attribution is not. Remove the trailer and
  recommit — you are the author of record.**

## PR process

- PRs are **squash-merged only**; head branches auto-delete.
- `required-green` (fmt, clippy, test, docs, deny, corpus) must pass, and the
  commit-policy gate checks DCO + attribution hygiene on every commit in the
  range.
- Keep the PR checklist honest — especially the line about the fourteen
  measured rows.
