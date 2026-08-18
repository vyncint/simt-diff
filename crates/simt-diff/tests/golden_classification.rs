//! Golden tests for the classification logic (brief §40).
//!
//! Each case states the evidence and the one classification it must produce.
//! These run with no GPU, no reconverge, and no network.

use std::collections::BTreeMap;

use simt_diff::classify::{Classification, Evidence, classify};
use simt_diff::oracle::ConstructionOracle;
use simt_diff::records::*;

fn generator(oracle: ConstructionOracle, model: Option<ReferenceModel>) -> GeneratorRecord {
    GeneratorRecord {
        template_id: "t".into(),
        generator_version: "test".into(),
        seed: 1,
        oracle,
        oracle_reason: "fixture".into(),
        kernel_source: "fn k() {}".into(),
        kernel_sha256: "0".repeat(64),
        kernel_name: "probe".into(),
        launches: vec![Launch::one_block(32)],
        reference_model: model,
        documented_limitation: None,
        expected_static: simt_diff::prediction::ExpectedStatic::Unspecified,
        prediction_basis: None,
    }
}

fn analyzer(findings: Vec<(&str, Confidence)>) -> AnalyzerRecord {
    AnalyzerRecord {
        tool: "reconverge".into(),
        version: "0.1.6".into(),
        command: vec!["cargo".into(), "reconverge".into(), "check".into()],
        exit_code: Some(if findings.iter().any(|(_, c)| c.gates()) { 1 } else { 0 }),
        findings: findings
            .into_iter()
            .map(|(code, confidence)| Finding {
                code: code.into(),
                confidence,
                message: "m".into(),
                kernel: Some("probe".into()),
                notes: vec![],
            })
            .collect(),
        raw_stdout: String::new(),
        raw_stderr: String::new(),
        witnesses: vec![],
        crashed: false,
        timed_out: false,
    }
}

fn run(outcome: RunOutcome, observed: Option<BTreeMap<u32, u32>>) -> GpuRunRecord {
    GpuRunRecord {
        launch: Launch::one_block(32),
        command: vec!["./probe".into(), "32".into()],
        outcome,
        exit_code: Some(0),
        seconds: 0.4,
        watchdog_seconds: 20,
        stdout: String::new(),
        stderr: String::new(),
        observed,
    }
}

fn sanitizer(reported: bool) -> SanitizerRecord {
    SanitizerRecord {
        tool: "synccheck".into(),
        command: vec!["compute-sanitizer".into()],
        launch: Launch::one_block(32),
        reported,
        error_count: Some(if reported { 1 } else { 0 }),
        outcome: RunOutcome::Completed,
        raw: "========= ERROR SUMMARY: 0 errors".into(),
    }
}

fn model(expected: u32) -> ReferenceModel {
    ReferenceModel {
        description: format!("every lane observes {expected:#010x}"),
        expected: (0..32).map(|l| (l, expected)).collect(),
        launch: Launch::one_block(32),
    }
}

fn lanes(value: u32) -> BTreeMap<u32, u32> {
    (0..32).map(|l| (l, value)).collect()
}

fn verdict(
    g: &GeneratorRecord,
    a: &AnalyzerRecord,
    r: &[GpuRunRecord],
    s: &[SanitizerRecord],
) -> Classification {
    classify(&Evidence { generator: g, analyzer: a, runs: r, sanitizer: s }).classification
}

// ---- the four cases the brief's §46 requires ---------------------------

#[test]
fn known_safe_clean_analyzer_clean_gpu_is_agreement() {
    let g = generator(ConstructionOracle::KnownSafe, Some(model(0xffff_ffff)));
    let a = analyzer(vec![]);
    let r = [run(RunOutcome::Completed, Some(lanes(0xffff_ffff)))];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(false)]),
        Classification::AgreementSafe
    );
}

#[test]
fn known_unsafe_confirmed_finding_with_sanitizer_is_agreement_bug() {
    let g = generator(ConstructionOracle::KnownUnsafe, None);
    let a = analyzer(vec![("RC001", Confidence::Confirmed)]);
    let r = [run(RunOutcome::Completed, None)];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(true)]),
        Classification::ConfirmedStaticBugDynamicObserved
    );
}

#[test]
fn known_unsafe_silent_analyzer_with_sanitizer_is_potential_false_negative() {
    let g = generator(ConstructionOracle::KnownUnsafe, None);
    let a = analyzer(vec![]);
    let r = [run(RunOutcome::Completed, None)];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(true)]),
        Classification::PotentialFalseNegative
    );
}

#[test]
fn known_safe_with_gating_finding_is_potential_false_positive() {
    let g = generator(ConstructionOracle::KnownSafe, Some(model(0xffff_ffff)));
    let a = analyzer(vec![("RC002", Confidence::Confirmed)]);
    let r = [run(RunOutcome::Completed, Some(lanes(0xffff_ffff)))];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(false)]),
        Classification::PotentialFalsePositive
    );
}

// ---- the rules the measurements forced --------------------------------

#[test]
fn a_completing_gpu_never_argues_against_a_confirmed_finding() {
    // Baseline §9.3: a divergent sync_threads() completes on sm_86 with the
    // barrier still inside the branch in the emitted PTX. Completion must
    // not weaken the static finding.
    let g = generator(ConstructionOracle::KnownUnsafe, None);
    let a = analyzer(vec![("RC001", Confidence::Confirmed)]);
    let r = [
        run(RunOutcome::Completed, None),
        run(RunOutcome::Completed, None),
    ];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(false)]),
        Classification::AgreementBug
    );
}

#[test]
fn a_warning_tier_finding_can_never_be_a_false_positive() {
    // Baseline §2.1: a warning is not an assertion and does not gate.
    let g = generator(ConstructionOracle::KnownSafe, Some(model(1)));
    let a = analyzer(vec![("RC001", Confidence::Warning)]);
    let r = [run(RunOutcome::Completed, Some(lanes(1)))];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(false)]),
        Classification::AgreementSafe
    );
}

#[test]
fn a_warning_on_an_unsafe_kernel_is_unsupported_not_a_miss() {
    // The documented interprocedural / lane-environment behaviour: seen,
    // but never witness-promoted (baseline §2.4).
    let g = generator(ConstructionOracle::KnownUnsafe, None);
    let a = analyzer(vec![("RC001", Confidence::Warning)]);
    let r = [run(RunOutcome::Completed, None)];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(true)]),
        Classification::AnalyzerUnsupported
    );
}

#[test]
fn silent_analyzer_with_no_corroboration_is_inconclusive_not_a_miss() {
    // This is the shape the baseline measured for RC001 on sm_86: silence
    // everywhere. It must not be promoted to a false negative.
    let g = generator(ConstructionOracle::KnownUnsafe, None);
    let a = analyzer(vec![]);
    let r = [run(RunOutcome::Completed, None)];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(false)]),
        Classification::DynamicInconclusive
    );
}

#[test]
fn the_invalid_mask_is_caught_only_by_the_reference_model() {
    // Baseline §9.5: ballot_sync(0x0000ffff) with 32 lanes present returns
    // 0xffffffff -- byte-identical to the valid case. The value comparison
    // is the only source that sees it.
    let g = generator(ConstructionOracle::KnownMaskInvalid, Some(model(0x0000_ffff)));
    let a = analyzer(vec![]);
    let r = [run(RunOutcome::Completed, Some(lanes(0xffff_ffff)))];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(false)]),
        Classification::PotentialFalseNegative
    );
}

#[test]
fn without_a_reference_model_the_same_case_is_only_inconclusive() {
    // The contrapositive of the test above, and the reason every collective
    // template must ship a reference model (baseline §10.2).
    let g = generator(ConstructionOracle::KnownMaskInvalid, None);
    let a = analyzer(vec![]);
    let r = [run(RunOutcome::Completed, Some(lanes(0xffff_ffff)))];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(false)]),
        Classification::DynamicInconclusive
    );
}

#[test]
fn a_safe_template_whose_values_are_wrong_blames_the_template() {
    let g = generator(ConstructionOracle::KnownSafe, Some(model(0x0000_ffff)));
    let a = analyzer(vec![]);
    let r = [run(RunOutcome::Completed, Some(lanes(0xdead_beef)))];
    assert_eq!(
        verdict(&g, &a, &r, &[]),
        Classification::ConstructionOracleConflict
    );
}

#[test]
fn instrumentation_disagreement_is_never_blamed_on_the_analyzer() {
    let g = generator(ConstructionOracle::KnownUnsafe, None);
    let a = analyzer(vec![("RC001", Confidence::Confirmed)]);
    let r = [run(RunOutcome::Completed, None)];
    let s = [SanitizerRecord {
        outcome: RunOutcome::WatchdogFired,
        reported: false,
        ..sanitizer(false)
    }];
    assert_eq!(verdict(&g, &a, &r, &s), Classification::InstrumentationConflict);
}

#[test]
fn an_analyzer_crash_is_the_finding_and_outranks_everything_else() {
    let g = generator(ConstructionOracle::KnownUnsafe, None);
    let mut a = analyzer(vec![]);
    a.crashed = true;
    let r = [run(RunOutcome::WatchdogFired, None)];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(true)]),
        Classification::AnalyzerError
    );
}

#[test]
fn a_mutation_without_an_oracle_yields_no_verdict() {
    let g = generator(ConstructionOracle::NoOracle, None);
    let a = analyzer(vec![("RC001", Confidence::Confirmed)]);
    let r = [run(RunOutcome::Completed, None)];
    assert_eq!(verdict(&g, &a, &r, &[]), Classification::NoOracleAvailable);
}

#[test]
fn every_result_says_what_it_does_not_claim() {
    let g = generator(ConstructionOracle::KnownUnsafe, None);
    let a = analyzer(vec![("RC001", Confidence::Confirmed)]);
    let r = [run(RunOutcome::Completed, None)];
    let out = classify(&Evidence {
        generator: &g,
        analyzer: &a,
        runs: &r,
        sanitizer: &[],
    });
    assert!(!out.observed.is_empty(), "observations must be recorded");
    assert!(!out.interpretation.is_empty(), "reasoning must be recorded");
    assert!(
        !out.not_claimed.is_empty(),
        "every verdict must state its limits (brief §18)"
    );
}

// ---- documented limitations are not bugs (brief §33) -------------------

#[test]
fn a_documented_limitation_is_unsupported_not_a_false_negative() {
    // The shrunk-mask class is published at 0% recall in reconverge's own
    // conformance corpus. Filing it as a false negative would be filing a
    // bug against a stated limitation.
    let mut g = generator(ConstructionOracle::KnownMaskInvalid, Some(model(0x0000_ffff)));
    g.documented_limitation = Some("MUTATION.md shrinkmask row: expected recall 0".into());
    let a = analyzer(vec![]);
    let r = [run(RunOutcome::Completed, Some(lanes(0xffff_ffff)))];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(false)]),
        Classification::AnalyzerUnsupported
    );
}

#[test]
fn without_the_limitation_the_same_evidence_is_a_false_negative() {
    // The contrapositive: the limitation, not the evidence, is what makes
    // the difference -- so it must be declared per template, deliberately.
    let g = generator(ConstructionOracle::KnownMaskInvalid, Some(model(0x0000_ffff)));
    let a = analyzer(vec![]);
    let r = [run(RunOutcome::Completed, Some(lanes(0xffff_ffff)))];
    assert_eq!(
        verdict(&g, &a, &r, &[sanitizer(false)]),
        Classification::PotentialFalseNegative
    );
}

#[test]
fn repeated_identical_mismatches_are_collapsed_into_one_line() {
    let g = generator(ConstructionOracle::KnownMaskInvalid, Some(model(0x0000_ffff)));
    let a = analyzer(vec![]);
    let r = [run(RunOutcome::Completed, Some(lanes(0xffff_ffff)))];
    let out = classify(&Evidence { generator: &g, analyzer: &a, runs: &r, sanitizer: &[] });
    let value_lines = out.observed.iter().filter(|o| o.starts_with("value:")).count();
    assert_eq!(value_lines, 1, "32 identical lines is noise, not evidence");
    assert!(
        out.observed.iter().any(|o| o.contains("0..=31")),
        "the collapsed line should name the lane range: {:?}",
        out.observed
    );
}

#[test]
fn a_safe_kernel_the_analyzer_is_silent_about_is_agreement_even_with_no_gpu_run() {
    // The static half of this laboratory runs on a laptop, and most of what it
    // produces is this shape. Calling it inconclusive would report every clean
    // case in a GPU-less sweep as an open question -- while the same kernel with
    // a *warning* on it was already being called agreement, which was
    // inconsistent.
    let g = generator(ConstructionOracle::KnownSafe, None);
    let a = analyzer(vec![]);
    assert_eq!(verdict(&g, &a, &[], &[]), Classification::AgreementSafe);
}

#[test]
fn an_unsafe_kernel_nothing_corroborates_stays_inconclusive_with_no_gpu_run() {
    // The contrast that makes the rule above safe: here the open question really
    // is the dynamic one, so silence plus no run resolves nothing.
    let g = generator(ConstructionOracle::KnownUnsafe, None);
    let a = analyzer(vec![]);
    assert_eq!(verdict(&g, &a, &[], &[]), Classification::DynamicInconclusive);
}
