# Development workflows. Run `just setup` once per clone; keep `just ci`
# green before every push — never push red.

# materialize the pinned toolchain and wire the repo-local git hooks
setup:
    rustup toolchain install
    rustup show
    git config core.hooksPath .githooks

fmt:
    cargo fmt --all

test:
    cargo test --workspace

# wire the repo-local git hooks (commit-msg soft guard)
hooks:
    git config core.hooksPath .githooks

# everything CI gates on, locally
ci:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace
    cargo deny check
    ./scripts/check-corpus.sh

# analyzer drift: rebuild every corpus entry and re-run reconverge on it.
# Not in `ci` because it needs a built cargo-reconverge on the pinned nightly;
# set SIMT_DIFF_RECONVERGE to point at it.
regress:
    cargo run -p simt-diff -- regress

# the generated corpus and what the model predicts for it, no analyzer needed
mutants:
    cargo run -p simt-diff -- mutate --depth 1
