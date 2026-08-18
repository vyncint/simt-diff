//! The differential comparison engine.
//!
//! Reads the evidence records and produces one classification plus the
//! reasons for it. Two rules are enforced here rather than documented,
//! because both were learned the expensive way:
//!
//! 1. **A clean GPU run never argues against a static finding.** The
//!    baseline's §9.3 measured a divergent `sync_threads()` completing
//!    normally on sm_86, with the barrier provably still inside the branch
//!    in the emitted PTX. A laboratory that treated completion as evidence
//!    of safety would call the analyzer's flagship diagnostic a false
//!    positive, on hardware, repeatably, and be wrong every time.
//! 2. **Only a gating-tier finding can be a false positive.** A
//!    `warning` is not an assertion (baseline §2.1); reporting one as a
//!    false positive would be reporting the tool for working as specified.

use serde::{Deserialize, Serialize};

use crate::oracle::{ConstructionOracle, OracleStrength};
use crate::records::{AnalyzerRecord, GeneratorRecord, GpuRunRecord, RunOutcome, SanitizerRecord};

/// Codes this laboratory reasons about. RC003/4/5 are syntactic or capacity
/// checks with no interesting dynamic counterpart (baseline §6).
pub const CONVERGENCE_CODES: &[&str] = &["RC001", "RC002"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum Classification {
    AgreementSafe,
    AgreementBug,
    ConfirmedStaticBugDynamicObserved,
    PotentialFalseNegative,
    PotentialFalsePositive,
    ConstructionOracleConflict,
    DynamicInconclusive,
    AnalyzerUnsupported,
    AnalyzerError,
    AnalyzerTimeout,
    GpuCompileError,
    GpuLaunchError,
    GpuTimeout,
    InstrumentationConflict,
    NondeterministicObservation,
    InfrastructureFailure,
    NoOracleAvailable,
}

/// The verdict, with its evidence kept attached. A human must be able to see
/// why the classification happened (brief §17).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DifferentialResult {
    pub schema: String,
    pub classification: Classification,
    pub oracle: ConstructionOracle,
    /// Which evidence sources actually spoke, for this case.
    pub strengths: Vec<OracleStrength>,
    /// Plain observations, no interpretation. Brief §18.
    pub observed: Vec<String>,
    /// Why those observations produce this classification.
    pub interpretation: Vec<String>,
    /// What this case explicitly does *not* establish.
    pub not_claimed: Vec<String>,
}

/// The evidence available for one case.
pub struct Evidence<'a> {
    pub generator: &'a GeneratorRecord,
    pub analyzer: &'a AnalyzerRecord,
    pub runs: &'a [GpuRunRecord],
    pub sanitizer: &'a [SanitizerRecord],
}

impl Evidence<'_> {
    fn any_watchdog(&self) -> bool {
        self.runs
            .iter()
            .any(|r| r.outcome == RunOutcome::WatchdogFired)
    }

    fn all_completed(&self) -> bool {
        !self.runs.is_empty() && self.runs.iter().all(|r| r.outcome == RunOutcome::Completed)
    }

    fn sanitizer_reported(&self) -> bool {
        self.sanitizer.iter().any(|s| s.reported)
    }

    /// Lanes whose observed value contradicts the template's reference
    /// model, across every run that reported values.
    ///
    /// Identical mismatches are collapsed into one line per (expected,
    /// observed) pair: 32 copies of the same sentence is not evidence, it is
    /// noise, and a reader has to be able to see the shape of the
    /// disagreement at a glance.
    fn reference_mismatches(&self) -> Vec<String> {
        let Some(model) = &self.generator.reference_model else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for run in self.runs {
            let Some(observed) = &run.observed else {
                continue;
            };
            if run.launch != model.launch {
                continue;
            }
            let mut grouped: std::collections::BTreeMap<(u32, Option<u32>), Vec<u32>> =
                std::collections::BTreeMap::new();
            for (lane, expected) in &model.expected {
                let got = observed.get(lane).copied();
                if got != Some(*expected) {
                    grouped.entry((*expected, got)).or_default().push(*lane);
                }
            }
            for ((expected, got), lanes) in grouped {
                let what = match got {
                    Some(g) => format!("observed {g:#010x}"),
                    None => "no value reported".to_string(),
                };
                out.push(format!(
                    "block={}: {} lane(s) {} expected {expected:#010x}, {what} \
                     -- reference model: {}",
                    run.launch.block.0,
                    lanes.len(),
                    describe_lanes(&lanes),
                    model.description
                ));
            }
        }
        out
    }
}

/// `0..=31` where contiguous, an explicit list when not.
fn describe_lanes(lanes: &[u32]) -> String {
    if lanes.is_empty() {
        return "(none)".to_string();
    }
    let contiguous = lanes.windows(2).all(|w| w[1] == w[0] + 1);
    if contiguous && lanes.len() > 2 {
        format!("{}..={}", lanes[0], lanes[lanes.len() - 1])
    } else {
        lanes
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Classify one case from its evidence.
pub fn classify(ev: &Evidence<'_>) -> DifferentialResult {
    let oracle = ev.generator.oracle;
    let gating = ev.analyzer.gating(CONVERGENCE_CODES);
    let warnings = ev.analyzer.warnings(CONVERGENCE_CODES);
    let mismatches = ev.reference_mismatches();

    let mut observed = Vec::new();
    let mut interpretation = Vec::new();
    let mut not_claimed = Vec::new();
    let mut strengths = Vec::new();

    if oracle != ConstructionOracle::NoOracle {
        strengths.push(OracleStrength::Construction);
    }
    observed.push(format!(
        "construction: {oracle:?} -- {}",
        ev.generator.oracle_reason
    ));
    observed.push(match (gating.len(), warnings.len()) {
        (0, 0) => "static: no RC001/RC002 finding at any tier".to_string(),
        (0, w) => format!("static: {w} finding(s) at warning tier only"),
        (g, w) => format!("static: {g} gating finding(s), {w} warning(s)"),
    });
    for run in ev.runs {
        observed.push(format!(
            "gpu: block={} -> {:?} in {:.2}s (watchdog {}s)",
            run.launch.block.0, run.outcome, run.seconds, run.watchdog_seconds
        ));
    }
    for s in ev.sanitizer {
        observed.push(format!(
            "{}: {} (errors: {})",
            s.tool,
            if s.reported { "reported" } else { "clean" },
            s.error_count.map_or("unknown".into(), |c| c.to_string())
        ));
    }
    for m in &mismatches {
        observed.push(format!("value: {m}"));
    }

    if !ev.runs.is_empty() {
        strengths.push(OracleStrength::DynamicObserved);
    }
    if ev.sanitizer_reported() {
        strengths.push(OracleStrength::SanitizerObserved);
    }

    // ---- infrastructure and tool failures come first: they invalidate the
    // ---- rest of the evidence rather than competing with it.
    if ev.analyzer.timed_out {
        return done(
            Classification::AnalyzerTimeout,
            oracle,
            strengths,
            observed,
            vec!["the analyzer did not finish; no static evidence exists".into()],
            vec!["nothing about the kernel's convergence properties".into()],
        );
    }
    if ev.analyzer.crashed {
        return done(
            Classification::AnalyzerError,
            oracle,
            strengths,
            observed,
            vec!["the analyzer exited abnormally; that is itself the finding".into()],
            vec!["nothing about the kernel's convergence properties".into()],
        );
    }
    if ev
        .runs
        .iter()
        .any(|r| r.outcome == RunOutcome::CompileFailed)
    {
        return done(
            Classification::GpuCompileError,
            oracle,
            strengths,
            observed,
            vec!["the case did not compile for the device".into()],
            vec!["nothing about hardware behaviour".into()],
        );
    }
    if ev
        .runs
        .iter()
        .any(|r| r.outcome == RunOutcome::LaunchFailed)
    {
        return done(
            Classification::GpuLaunchError,
            oracle,
            strengths,
            observed,
            vec!["the launch was rejected".into()],
            vec!["nothing about the kernel body".into()],
        );
    }

    // ---- instrumentation must not be blamed on the analyzer (brief §14).
    let raw_completed = ev.all_completed();
    let sanitized_broke = ev.sanitizer.iter().any(|s| {
        matches!(
            s.outcome,
            RunOutcome::WatchdogFired | RunOutcome::NonzeroExit
        )
    });
    if raw_completed && sanitized_broke {
        return done(
            Classification::InstrumentationConflict,
            oracle,
            strengths,
            observed,
            vec![
                "the raw run completed while the sanitized run did not; the \
                  instrumentation changed the behaviour under test"
                    .into(),
            ],
            vec!["nothing about the analyzer".into()],
        );
    }

    if oracle == ConstructionOracle::NoOracle {
        return done(
            Classification::NoOracleAvailable,
            oracle,
            strengths,
            observed,
            vec![
                "a mutation invalidated the semantic label, so agreement is \
                  not defined for this case"
                    .into(),
            ],
            vec!["any claim about correctness in either direction".into()],
        );
    }
    if oracle == ConstructionOracle::KnownOutsideAnalyzerScope {
        return done(
            Classification::AnalyzerUnsupported,
            oracle,
            strengths,
            observed,
            vec![
                "the construct is outside the analyzer's documented surface, \
                  so silence is correct behaviour"
                    .into(),
            ],
            vec!["that the analyzer is wrong".into()],
        );
    }

    // ---- the construction oracle disagreeing with hardware means the
    // ---- template is suspect first, never the analyzer (brief §6).
    if oracle.asserts_valid() && !mismatches.is_empty() {
        interpretation.push(
            "a safe-by-construction template produced values its own reference \
             model rejects; the template is wrong, or the reference model is"
                .into(),
        );
        not_claimed.push("any conclusion about the analyzer".into());
        return done(
            Classification::ConstructionOracleConflict,
            oracle,
            strengths,
            observed,
            interpretation,
            not_claimed,
        );
    }

    // ---- unsafe by construction ------------------------------------------
    if oracle.asserts_invalid() {
        let dynamic_evidence =
            !mismatches.is_empty() || ev.sanitizer_reported() || ev.any_watchdog();

        if !gating.is_empty() {
            interpretation.push(
                "the analyzer asserts the bug at a gating tier and construction \
                 agrees"
                    .into(),
            );
            not_claimed.push(
                "that every launch of this kernel misbehaves; undefined \
                 behaviour need not manifest"
                    .into(),
            );
            let class = if dynamic_evidence {
                interpretation.push("independent dynamic evidence matches the prediction".into());
                Classification::ConfirmedStaticBugDynamicObserved
            } else {
                interpretation.push(
                    "the hardware completed anyway, which is expected for \
                     undefined behaviour and is not counter-evidence \
                     (baseline §9.3)"
                        .into(),
                );
                Classification::AgreementBug
            };
            return done(
                class,
                oracle,
                strengths,
                observed,
                interpretation,
                not_claimed,
            );
        }

        // No gating finding. Whether this is interesting depends on whether a
        // documented limitation already predicts the silence.
        if !warnings.is_empty() {
            interpretation.push(
                "the analyzer saw it but declined to promote it past warning \
                 tier; that is the documented behaviour for constructs it \
                 cannot witness-replay, not a miss"
                    .into(),
            );
            not_claimed.push("that this is a false negative".into());
            return done(
                Classification::AnalyzerUnsupported,
                oracle,
                strengths,
                observed,
                interpretation,
                not_claimed,
            );
        }

        if let Some(limitation) = &ev.generator.documented_limitation {
            interpretation.push(format!(
                "the analyzer documents this class as outside its current \
                 scope, so silence is specified behaviour, not a miss: {limitation}"
            ));
            if dynamic_evidence {
                interpretation.push(
                    "the dynamic evidence is still recorded, because it says \
                     something the static gap does not: what hardware actually \
                     does when nothing catches the bug"
                        .into(),
                );
            }
            not_claimed.push(
                "that this is a false negative; a documented limitation is the \
                 tool working as specified"
                    .into(),
            );
            return done(
                Classification::AnalyzerUnsupported,
                oracle,
                strengths,
                observed,
                interpretation,
                not_claimed,
            );
        }

        if dynamic_evidence {
            interpretation.push(
                "construction says the program is invalid, independent dynamic \
                 evidence agrees, and the analyzer reported nothing at any \
                 tier"
                    .into(),
            );
            not_claimed.push(
                "that the analyzer is required to catch it -- the case must be \
                 checked against the documented surface before filing"
                    .into(),
            );
            return done(
                Classification::PotentialFalseNegative,
                oracle,
                strengths,
                observed,
                interpretation,
                not_claimed,
            );
        }

        interpretation.push(
            "construction says invalid and the analyzer is silent, but nothing \
             independent corroborates it: no value mismatch, no sanitizer \
             report, no watchdog. On this stack that is the normal outcome for \
             a divergent barrier (baseline §9.3/§9.4)"
                .into(),
        );
        not_claimed.push("a false negative; there is no second source".into());
        return done(
            Classification::DynamicInconclusive,
            oracle,
            strengths,
            observed,
            interpretation,
            not_claimed,
        );
    }

    // ---- safe by construction --------------------------------------------
    if !gating.is_empty() {
        interpretation.push(
            "the analyzer asserts a bug at a gating tier in a kernel that is \
             safe by construction"
                .into(),
        );
        if ev.all_completed() && mismatches.is_empty() {
            interpretation.push(
                "every generated launch completed with values matching the \
                 reference model"
                    .into(),
            );
        }
        not_claimed.push(
            "that the kernel is safe for every launch; only the generated \
             finite domain was run"
                .into(),
        );
        return done(
            Classification::PotentialFalsePositive,
            oracle,
            strengths,
            observed,
            interpretation,
            not_claimed,
        );
    }
    if !warnings.is_empty() {
        interpretation.push(
            "warning-tier findings on a safe kernel are not assertions and do \
             not gate; this is within specification"
                .into(),
        );
        return done(
            Classification::AgreementSafe,
            oracle,
            strengths,
            observed,
            interpretation,
            not_claimed,
        );
    }
    // A static-only sweep can still settle this branch: construction says the
    // kernel is fine and the analyzer agrees, which is agreement whether or not
    // a GPU was involved. `DynamicInconclusive` is for the case above, where the
    // open question *is* the dynamic one -- construction says invalid and only a
    // run could corroborate it. Answering "inconclusive" here instead would have
    // labelled every clean case in a laptop sweep as unresolved.
    if ev.runs.is_empty() {
        interpretation.push(
            "construction and the analyzer agree the kernel is fine; no launch \
             was executed, and none is needed to say that"
                .into(),
        );
        not_claimed.push(
            "anything about hardware behaviour, or about launches outside the \
             one analyzed"
                .into(),
        );
        return done(
            Classification::AgreementSafe,
            oracle,
            strengths,
            observed,
            interpretation,
            not_claimed,
        );
    }
    interpretation.push("construction, the analyzer, and every generated launch agree".into());
    not_claimed.push("safety for launches outside the generated domain".into());
    done(
        Classification::AgreementSafe,
        oracle,
        strengths,
        observed,
        interpretation,
        not_claimed,
    )
}

fn done(
    classification: Classification,
    oracle: ConstructionOracle,
    mut strengths: Vec<OracleStrength>,
    observed: Vec<String>,
    interpretation: Vec<String>,
    not_claimed: Vec<String>,
) -> DifferentialResult {
    strengths.sort_unstable();
    strengths.dedup();
    DifferentialResult {
        schema: crate::records::SCHEMA_DIFFERENTIAL.to_string(),
        classification,
        oracle,
        strengths,
        observed,
        interpretation,
        not_claimed,
    }
}
