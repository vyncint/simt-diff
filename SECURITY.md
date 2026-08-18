# Security Policy

## Supported versions

simt-diff is a research tool in pre-release development: there are no tagged
releases and nothing is published to crates.io. The supported version is the tip
of `main`.

## Reporting a vulnerability

Report privately — please do not open a public issue:

- GitHub → **Security** → **Report a vulnerability** (preferred), or
- directly to the maintainer, [@vyncint](https://github.com/vyncint).

Please include a minimal reproducer and what you expected instead.

## Threat model

Three properties are worth stating plainly, because they shape what counts as a
vulnerability here.

**This tool generates programs and then compiles them.** `simt-diff` writes
crates and invokes `cargo` on them through the analyzer under test. Those crates
depend on the device library being studied, so build scripts and proc macros in
that dependency tree execute exactly as they would under `cargo check`. It
inherits cargo's trust model and adds no sandbox: **do not point it at a
dependency revision you would not be willing to build.** This is expected
behavior, not a vulnerability.

**Generated cases are adversarial by construction.** Every kernel in `cases/`
and `corpus/` is designed to be wrong in a specific way — divergent barriers,
invalid warp masks, undefined behaviour. They are inputs to an analyzer, not
code to run in production. The runner crates are emitted so they *can* be
executed on a GPU deliberately, under supervision, on hardware you are willing
to hang.

**Findings here are claims about a third-party analyzer.** A wrong claim in this
repository is a correctness bug, not a security issue — file it with the
"Finding no longer reproduces" form. A security issue in
[reconverge](https://github.com/vyncint/reconverge) belongs in its tracker,
under its policy.
