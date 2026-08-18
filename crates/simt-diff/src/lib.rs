//! simt-diff — a differential laboratory for SIMT static analyzers.
//!
//! The design of this crate is driven by measurements recorded in
//! `docs/research-baseline.md`. Three of them matter enough to restate:
//!
//! - A divergent `sync_threads()` **completes** on sm_86, with the barrier
//!   provably still inside the branch in the emitted PTX (§9.3). Process
//!   completion is therefore not evidence of safety, and a watchdog is not
//!   the dynamic signal for RC001.
//! - `compute-sanitizer synccheck` reported nothing for either divergent
//!   shape (§9.4), so the vendor dynamic checker is a weaker oracle here
//!   than one would expect.
//! - An invalid warp mask returned a value byte-identical to the valid
//!   case (§9.5). Only construction knowledge distinguished them, which is
//!   why every collective template carries a reference model.

#![forbid(unsafe_code)]

pub mod analyzer;
pub mod classify;
pub mod emit;
pub mod oracle;
pub mod records;
pub mod runner_output;
pub mod templates;
