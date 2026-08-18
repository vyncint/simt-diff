## Summary

<!-- What does this change, and why? -->

## Checklist

- [ ] Title is a scoped Conventional Commit (e.g. `model: …`), imperative, ≤ 72 chars
- [ ] Every commit is signed off (`git commit -s`; the DCO trailer matches the author)
- [ ] No AI attribution anywhere (trailers, message bodies, identities)
- [ ] Tests added or updated for the change
- [ ] `tests/ir_seeds.rs` still reproduces the fourteen measured Stage 3 rows — or the
      change to a measurement is called out in the summary
- [ ] Any new or changed model rule states its **provenance** (quoted / extrapolated /
      measured) and, if measured, names the case that established it
- [ ] `scripts/check-corpus.sh` passes — or a re-recorded corpus entry is justified here
