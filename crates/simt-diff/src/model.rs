//! A falsifiable model of reconverge, derived from its documentation.
//!
//! Stage 3 declared one prediction per hand-written template. That does not
//! scale to generated kernels, and hand-declaring a prediction for a mutant
//! would be the same sin as hand-declaring its oracle. So the rules are
//! written out once, here, as a function of a kernel's *static* features -- and
//! the fourteen measured Stage 3 rows become the test that the function is
//! right (`tests/ir_seeds.rs`).
//!
//! The distinction that makes this honest is [`Provenance`]:
//!
//! - [`Provenance::Quoted`] -- the documentation states the behaviour for this
//!   construct. A violation is a statement about reconverge.
//! - [`Provenance::Extrapolated`] -- the documentation states a *reason*, and I
//!   inferred what it implies here. A violation is, first, a statement about
//!   this model. Reporting one as an analyzer bug would be dishonest.
//!
//! Static features only. The one place the model consults executed facts is
//! where reconverge's own mechanism does: the witness interpreter replays 32
//! lanes, so a guard it can evaluate and finds uniform gives it nothing to
//! promote. Modelling that is modelling the documented mechanism, not peeking.

use serde::{Deserialize, Serialize};

use crate::interpret::{Semantics, Site, SiteKind};
use crate::ir::Mask;
use crate::prediction::ExpectedStatic;

/// Where a prediction comes from, and therefore what a violation means.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Provenance {
    /// The documentation states this behaviour for this construct.
    Quoted { source: String },
    /// Inferred from a documented reason. A violation indicts the model first.
    Extrapolated { basis: String },
    /// Not documented anywhere, but measured by this laboratory: the rule was
    /// written from an observed run, and the case that established it is named.
    /// A violation of one of these is a *regression*, which is a different and
    /// more actionable claim than either of the other two.
    Measured { evidence: String },
}

impl Provenance {
    pub fn is_quoted(&self) -> bool {
        matches!(self, Provenance::Quoted { .. })
    }

    pub fn source(&self) -> &str {
        match self {
            Provenance::Quoted { source } => source,
            Provenance::Extrapolated { basis } => basis,
            Provenance::Measured { evidence } => evidence,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Provenance::Quoted { .. } => "quoted",
            Provenance::Extrapolated { .. } => "extrapolated",
            Provenance::Measured { .. } => "measured",
        }
    }
}

/// What the model says will happen, why, and how much the "why" is worth.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPrediction {
    pub expected: ExpectedStatic,
    /// Stable identifier for the rule that fired, so a corpus can be grouped
    /// by rule and a regression can name one.
    pub rule: String,
    pub provenance: Provenance,
    /// The analyzer's own words about not covering this class, when the rule
    /// rests on one. Feeds `GeneratorRecord::documented_limitation`, which is
    /// what keeps the classifier from calling a published gap a false negative.
    pub documented_limitation: Option<String>,
}

const INTERPROCEDURAL: &str = "README Limitations: interprocedural analysis is summary-based in v1 \
    (per-function may_contain_barrier bits, no context sensitivity); call-site \
    findings stay at warning and are never witness-promoted";
const LANE_ENVIRONMENT: &str = "README Limitations: guards built on the lane-environment registers \
    stay warnings -- the witness interpreter cannot yet evaluate their values \
    (needs width-typed integer ! and truncating casts)";
const NON_LITERAL_MASK: &str = "README Limitations: masks that are not literals -- a named const, \
    or anything computed -- cannot be evaluated through rustc_public at the \
    pinned toolchain";
const UNMASKED_WRAPPER: &str = "dialect simt.rs test `unmasked_wrappers_are_the_documented_v1_gap`: \
    warp::shuffle, warp::ballot, warp::all/any and the reduce_* helpers hide an \
    implicit full mask inside cuda-device and are not analyzed in v1";
const SHRINKMASK: &str = "conformance/MUTATION.md, shrinkmask row: RC002 v1 checks convergence; it \
    does not do mask arithmetic against launch shapes -- expected recall 0 in v1";
const RC001_SURFACE: &str = "README RC001: a barrier reachable under thread-divergent control, \
    witness-replayed over 32 lanes and promoted to confirmed when a divergent \
    lane pair is found";
const PRECISION: &str = "README: zero false positives at default confidence is a requirement, not \
    a goal -- a kernel with nothing wrong in the analyzed surface must be silent";
const WITNESS_REPLAY: &str = "the witness interpreter replays 32 lanes; a guard it can evaluate and \
    finds uniform gives it no divergent lane pair to confirm";
const M_TRUNCATING_CAST: &str = "measured on reconverge 0.1.6, case \
    `barrier_divergent_intra_warp+truncate_operand@0`: a guard written \
    `(i.get() as u8) as u32 % 2 == 0` was reported RC001/confirmed with a witness \
    artifact. The witness interpreter does evaluate truncating casts, so the \
    README's stated reason for the lane-environment gap (\"needs width-typed \
    evaluation of integer ! and truncating casts\") does not extend to casts on \
    the thread index";
const M_MASK_ARITHMETIC: &str = "measured on reconverge 0.1.6: of 26 RC002 findings in the depth-1 \
    corpus, all 14 promoted to confirmed were calls where some lane the mask \
    names is absent, and none of the 12 held at warning tier were. \
    `collective_under_divergence+mask_match_participants@0.0` -- the correct \
    guarded partial-warp idiom, mask 0x5555_5555 under an even-lane guard -- was \
    reported at warning tier and never gated. RC002 therefore does compare the \
    mask against the lanes present, which conformance/MUTATION.md says v1 does \
    not do; the analyzer is more capable than its own documentation claims, and \
    the consequence is precision";
const M_GUARD_INSIDE_LOOP: &str = "measured on reconverge 0.1.6, minimized in the depth-2 sweep: \
    `while n < (2) { if i.get() % 2 == 0 { sync_threads() } }` was reported \
    RC001/warning with no witness, while `if i.get() % 2 == 0 { while n < (2) \
    { sync_threads() } }` -- the same two constructs, nested the other way -- was \
    RC001/confirmed with a witness. A loop inside the guard is replayed; a \
    divergent guard inside a loop is not, and the loop's own trip count does not \
    matter (uniform and thread-derived bounds behave identically). No \
    documentation mentions loops at all";
const M_ONE_WITNESSED_SOURCE: &str = "measured on reconverge 0.1.6. Three versions of this rule were \
    wrong before this one, and each was refuted by a case built to demonstrate \
    it. (a) Two barriers under evaluable guards give RC001/confirmed plus \
    RC001/warning, so the second is not promoted. (b) Making the *first* guard a \
    lane-environment read drops both to warning, while appending such a barrier \
    after a confirmable one changes nothing -- which killed \"the effect is \
    function-wide\". (c) Two barriers under *identical* guards are BOTH \
    confirmed, with two witness artifacts -- which killed \"only the first \
    finding is ever promoted\". (d) A held-out case, \
    `collective_under_divergence+mask_single_lane@0.0+complementary_guard@0`, \
    predicted gating and came back warning: its first collective is fine and its \
    second names an absent lane, and neither was promoted. What fits all of it: \
    the witness pass attempts the first divergence source in program order, \
    skipping call sites, and promotes exactly those findings that share that \
    source. An unevaluable first source (a lane-environment read, or a guard \
    inside a loop) leaves nothing in the function promotable. The mechanism \
    behind that choice is still not established; the discriminating experiment \
    is named in docs/stage-4.md";
const M_UNREACHABLE: &str = "measured on reconverge 0.1.6, case \
    `barrier_divergent_nested+invert_guard@0`: a barrier no thread of the \
    declared launch can reach was still reported RC001, at warning tier, with no \
    witness. The syntactic recognizer speaks and the replay declines to promote, \
    which is consistent and undocumented";

/// Rank so several sites can be combined: whichever site the analyzer would
/// treat most severely is the one the case's prediction is about.
fn severity(e: &ExpectedStatic) -> u8 {
    match e {
        ExpectedStatic::Gating { .. } => 3,
        ExpectedStatic::WarningOnly { .. } => 2,
        ExpectedStatic::Silent => 1,
        ExpectedStatic::Unspecified => 0,
    }
}

pub fn predict(sem: &Semantics) -> ModelPrediction {
    let mut best: Option<ModelPrediction> = None;
    // A downgrade is the informative thing to report when several sites tie at
    // warning tier: "this would have gated but for what precedes it" tells a
    // reader more than "the first site's guard is unevaluable".
    let mut downgraded: Option<ModelPrediction> = None;
    for site in &sem.sites {
        let mut p = if !site.executed() {
            unreachable_site(site)
        } else {
            match site.kind {
                SiteKind::Barrier => predict_barrier(site),
                SiteKind::Collective => predict_collective(site),
            }
        };

        // A site's own guard is not the whole story: only the divergence source
        // the witness pass actually attempted can be promoted.
        if let ExpectedStatic::Gating { code } = &p.expected
            && witnessed_source(sem) != site.guards.divergence_source
        {
            p = measured(
                ExpectedStatic::WarningOnly { code: code.clone() },
                "another_divergence_source_got_the_witness",
                M_ONE_WITNESSED_SOURCE,
                None,
            );
            downgraded = Some(p.clone());
        }

        let replace = match &best {
            None => true,
            Some(b) => severity(&p.expected) > severity(&b.expected),
        };
        if replace {
            best = Some(p);
        }
    }
    let best = best.unwrap_or_else(|| ModelPrediction {
        expected: ExpectedStatic::Silent,
        rule: "no_analyzed_construct".to_string(),
        provenance: Provenance::Quoted { source: PRECISION.to_string() },
        documented_limitation: None,
    });
    match downgraded {
        Some(d) if severity(&best.expected) <= severity(&d.expected) => d,
        _ => best,
    }
}

/// The divergence source the witness pass attempted, if anything can be promoted.
///
/// The first per-thread condition in program order, skipping call sites -- a
/// summary-based finding is never witness-replayed, so it is not an attempt.
/// `None` when that first source is one the interpreter cannot evaluate, which is
/// measured to leave nothing in the function promotable.
fn witnessed_source(sem: &Semantics) -> Option<String> {
    let first = sem.sites.iter().find(|s| {
        s.executed() && s.guards.statically_divergent && s.guards.via_helper_depth == 0
    })?;
    if first.guards.lane_env || first.guards.divergent_guard_inside_loop {
        return None;
    }
    first.guards.divergence_source.clone()
}

/// A construct present in the source that no thread of this launch reaches.
///
/// Silence would be defensible -- there is no divergent lane pair to find -- but
/// it is not what happens: the syntactic recognizer reports it and the replay
/// declines to promote it. That is a sensible split, since a launch contract is
/// a declaration and not a proof, and it appears in no documentation.
fn unreachable_site(site: &Site) -> ModelPrediction {
    // Only a guarded construct is recognized at all; an unconditional one that
    // is unreachable does not exist in the CFG the analyzer walks.
    if !site.guards.statically_divergent {
        return quoted(
            ExpectedStatic::Silent,
            "unreachable_construct_under_uniform_control",
            PRECISION,
            None,
        );
    }
    let code = match site.kind {
        SiteKind::Barrier => "RC001",
        SiteKind::Collective => "RC002",
    };
    measured(
        ExpectedStatic::WarningOnly { code: code.to_string() },
        "construct_unreachable_at_this_launch",
        M_UNREACHABLE,
        None,
    )
}

fn quoted(
    expected: ExpectedStatic,
    rule: &str,
    source: &str,
    limitation: Option<&str>,
) -> ModelPrediction {
    ModelPrediction {
        expected,
        rule: rule.to_string(),
        provenance: Provenance::Quoted { source: source.to_string() },
        documented_limitation: limitation.map(str::to_string),
    }
}

fn measured(
    expected: ExpectedStatic,
    rule: &str,
    evidence: &str,
    limitation: Option<&str>,
) -> ModelPrediction {
    ModelPrediction {
        expected,
        rule: rule.to_string(),
        provenance: Provenance::Measured { evidence: evidence.to_string() },
        documented_limitation: limitation.map(str::to_string),
    }
}

fn extrapolated(
    expected: ExpectedStatic,
    rule: &str,
    basis: &str,
    limitation: Option<&str>,
) -> ModelPrediction {
    ModelPrediction {
        expected,
        rule: rule.to_string(),
        provenance: Provenance::Extrapolated { basis: basis.to_string() },
        documented_limitation: limitation.map(str::to_string),
    }
}

fn rc001() -> ExpectedStatic {
    ExpectedStatic::Gating { code: "RC001".to_string() }
}

fn rc001_warning() -> ExpectedStatic {
    ExpectedStatic::WarningOnly { code: "RC001".to_string() }
}

fn predict_barrier(site: &Site) -> ModelPrediction {
    let g = &site.guards;

    // 1. Nothing per-thread encloses it. Silence is the requirement.
    if !g.statically_divergent {
        return quoted(ExpectedStatic::Silent, "barrier_uniform_control", PRECISION, None);
    }

    // 2. The barrier is behind a call. The documented rule is unconditional
    //    and outranks everything below it, including an evaluable guard at the
    //    call site -- which is exactly what makes it a rule worth testing.
    if g.via_helper_depth > 0 {
        return quoted(
            rc001_warning(),
            "barrier_via_call_site",
            INTERPROCEDURAL,
            Some(INTERPROCEDURAL),
        );
    }

    // 3. Only the lane environment establishes the divergence. Kept separate
    //    from rule 4 below, which reaches the same tier by a different route:
    //    the two were derived differently and a report should say which fired.
    if g.lane_env && !g.index_evaluable {
        return quoted(
            rc001_warning(),
            "barrier_under_lane_environment_guard",
            LANE_ENVIRONMENT,
            Some(LANE_ENVIRONMENT),
        );
    }

    // 4. A mixed chain: one predicate is evaluable, another is not. The first
    //    version of this rule predicted promotion, reasoning that the evaluable
    //    conjunct alone exhibits a divergent lane pair. Measurement said
    //    otherwise -- one unevaluable predicate anywhere in the chain holds the
    //    whole finding at warning tier -- and on rereading, "findings under such
    //    guards are never witness-promoted" says exactly that.
    if g.lane_env {
        return quoted(
            rc001_warning(),
            "barrier_under_mixed_guard",
            LANE_ENVIRONMENT,
            Some(LANE_ENVIRONMENT),
        );
    }

    // 5. A divergent guard nested inside a loop. Undocumented, minimized, and
    //    it outranks the cast rule below: the loop shape suppresses promotion
    //    even when everything in the guard is otherwise evaluable.
    if g.divergent_guard_inside_loop {
        return measured(
            rc001_warning(),
            "barrier_under_guard_inside_loop",
            M_GUARD_INSIDE_LOOP,
            None,
        );
    }

    // 6. A truncating cast on the thread index. The README names casts as
    //    missing machinery; measurement shows the witness interpreter handles
    //    them, so this rule now follows the measurement and not the README.
    if g.trunc_cast {
        return if site.divergent {
            measured(rc001(), "barrier_under_truncating_cast_guard", M_TRUNCATING_CAST, None)
        } else {
            extrapolated(
                ExpectedStatic::Silent,
                "barrier_under_truncating_cast_uniform_guard",
                WITNESS_REPLAY,
                None,
            )
        };
    }

    // 7. Plain thread-index arithmetic, which the witness interpreter can
    //    replay -- so what it finds decides the tier.
    if site.divergent {
        quoted(rc001(), "barrier_under_evaluable_divergent_guard", RC001_SURFACE, None)
    } else {
        extrapolated(
            ExpectedStatic::Silent,
            "barrier_under_evaluable_uniform_guard",
            WITNESS_REPLAY,
            None,
        )
    }
}

fn predict_collective(site: &Site) -> ModelPrediction {
    let g = &site.guards;
    let mask = site.mask.unwrap_or(Mask::Literal(0xffff_ffff));

    // The wrapper hides its mask inside cuda-device, so nothing about the call
    // site matters: it is not analyzed at all in v1.
    if mask == Mask::ImplicitWrapper {
        return quoted(
            ExpectedStatic::Silent,
            "collective_via_unmasked_wrapper",
            UNMASKED_WRAPPER,
            Some(UNMASKED_WRAPPER),
        );
    }

    // A convergent call site. RC002 checks convergence and does no mask
    // arithmetic, so it is silent here whatever the mask names -- including
    // when the mask is wrong, which is the documented shrinkmask gap.
    if !g.statically_divergent {
        let limitation = if site.mask_valid == Some(false) {
            Some(SHRINKMASK)
        } else if !mask.is_literal() {
            Some(NON_LITERAL_MASK)
        } else {
            None
        };
        return quoted(
            ExpectedStatic::Silent,
            "collective_at_convergent_call_site",
            SHRINKMASK,
            limitation,
        );
    }

    let rc002 = ExpectedStatic::Gating { code: "RC002".to_string() };
    let rc002_warning = ExpectedStatic::WarningOnly { code: "RC002".to_string() };

    // Non-convergent call site. Everything that blocks RC001's promotion blocks
    // RC002's as well, in the same order.
    if g.via_helper_depth > 0 {
        return quoted(
            rc002_warning,
            "collective_via_call_site",
            INTERPROCEDURAL,
            Some(INTERPROCEDURAL),
        );
    }
    if g.lane_env {
        return quoted(
            rc002_warning,
            "collective_under_lane_environment_guard",
            LANE_ENVIRONMENT,
            Some(LANE_ENVIRONMENT),
        );
    }
    if g.divergent_guard_inside_loop {
        return measured(
            rc002_warning,
            "collective_under_guard_inside_loop",
            M_GUARD_INSIDE_LOOP,
            None,
        );
    }
    if !mask.is_literal() {
        return quoted(
            rc002_warning,
            "collective_with_unevaluable_mask",
            NON_LITERAL_MASK,
            Some(NON_LITERAL_MASK),
        );
    }

    // And then the finding that Stage 4 exists to produce: whether RC002 gates
    // is decided by mask arithmetic against the lanes actually present, not by
    // convergence alone. A mask that names exactly the lanes that show up is
    // reported and never gated, so the correct guarded partial-warp idiom is not
    // a false positive -- which is what the documentation, read literally, would
    // have predicted it to be.
    if site.value_defined {
        measured(
            rc002_warning,
            "collective_with_every_named_lane_present",
            M_MASK_ARITHMETIC,
            None,
        )
    } else {
        measured(rc002, "collective_naming_an_absent_lane", M_MASK_ARITHMETIC, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpret::interpret;
    use crate::ir::{CmpOp, Kernel, Pred, Stmt, Value};
    use crate::records::Launch;

    fn predict_kernel(k: &Kernel) -> ModelPrediction {
        predict(&interpret(k, Launch::one_block(32)))
    }

    fn even() -> Pred {
        Pred::Cmp(Value::Rem(Box::new(Value::LaneIndex), 2), CmpOp::Eq, 0)
    }

    fn quarter_guard() -> Pred {
        Pred::Cmp(Value::Rem(Box::new(Value::LaneIndex), 4), CmpOp::Eq, 0)
    }

    #[test]
    fn an_uniform_barrier_is_predicted_silent_on_the_precision_requirement() {
        let p = predict_kernel(&Kernel::new(vec![Stmt::Barrier]));
        assert_eq!(p.expected, ExpectedStatic::Silent);
        assert!(p.provenance.is_quoted());
    }

    #[test]
    fn the_call_site_rule_outranks_an_evaluable_guard() {
        // Both the interprocedural rule and the RC001 surface could apply. The
        // documentation makes the call-site rule unconditional, so it wins --
        // and the Stage 3 measurement agreed.
        let k = Kernel::with_helper(
            vec![Stmt::If { pred: even(), body: vec![Stmt::CallHelper] }],
            1,
        );
        let p = predict_kernel(&k);
        assert_eq!(p.expected, ExpectedStatic::WarningOnly { code: "RC001".into() });
        assert_eq!(p.rule, "barrier_via_call_site");
        assert!(p.provenance.is_quoted());
        assert!(p.documented_limitation.is_some());
    }

    #[test]
    fn a_truncating_cast_guard_follows_the_measurement_not_the_readme() {
        // The README names truncating casts as machinery the witness
        // interpreter lacks. This case was generated, run, and came back
        // RC001/confirmed with a witness -- so the rule follows what was
        // measured, and says so.
        let k = Kernel::new(vec![Stmt::If {
            pred: Pred::Cmp(
                Value::Rem(Box::new(Value::TruncU8(Box::new(Value::LaneIndex))), 2),
                CmpOp::Eq,
                0,
            ),
            body: vec![Stmt::Barrier],
        }]);
        let p = predict_kernel(&k);
        assert_eq!(p.expected, ExpectedStatic::Gating { code: "RC001".into() });
        assert!(matches!(p.provenance, Provenance::Measured { .. }));
        assert!(p.provenance.source().contains("truncate_operand@0"), "the rule names its case");
    }

    #[test]
    fn a_mixed_guard_is_held_at_warning_tier_like_a_pure_lane_environment_one() {
        // First written as "the evaluable conjunct gives the replay a lane pair,
        // so it promotes". Measurement said no, and the documented sentence
        // ("findings under such guards are never witness-promoted") covers it.
        let k = Kernel::new(vec![Stmt::If {
            pred: Pred::And(
                Box::new(even()),
                Box::new(Pred::Cmp(Value::WarpId, CmpOp::Eq, 0)),
            ),
            body: vec![Stmt::Barrier],
        }]);
        let p = predict_kernel(&k);
        assert_eq!(p.expected, ExpectedStatic::WarningOnly { code: "RC001".into() });
        assert!(p.provenance.is_quoted());
    }

    #[test]
    fn an_unreachable_guarded_barrier_is_predicted_at_warning_tier() {
        // No thread of block=32 reaches this barrier: the outer guard admits
        // odd lanes and the inner one admits multiples of four.
        let k = Kernel::new(vec![Stmt::If {
            pred: Pred::Not(Box::new(even())),
            body: vec![Stmt::If { pred: quarter_guard(), body: vec![Stmt::Barrier] }],
        }]);
        let sem = interpret(&k, Launch::one_block(32));
        assert_eq!(sem.oracle, crate::oracle::ConstructionOracle::KnownSafe);
        let p = predict(&sem);
        assert_eq!(p.expected, ExpectedStatic::WarningOnly { code: "RC001".into() });
        assert!(matches!(p.provenance, Provenance::Measured { .. }));
    }

    #[test]
    fn a_statically_divergent_but_dynamically_uniform_guard_is_predicted_silent() {
        let k = Kernel::new(vec![Stmt::If {
            pred: Pred::Cmp(Value::Rem(Box::new(Value::LaneIndex), 1), CmpOp::Eq, 0),
            body: vec![Stmt::Barrier],
        }]);
        let p = predict_kernel(&k);
        assert_eq!(p.expected, ExpectedStatic::Silent);
        assert_eq!(p.rule, "barrier_under_evaluable_uniform_guard");
        assert!(!p.provenance.is_quoted());
    }

    #[test]
    fn the_strongest_site_decides_a_multi_site_kernel() {
        let k = Kernel::new(vec![
            Stmt::Ballot { mask: Mask::ActiveMask },
            Stmt::If { pred: even(), body: vec![Stmt::Barrier] },
        ]);
        let p = predict_kernel(&k);
        assert_eq!(p.expected, ExpectedStatic::Gating { code: "RC001".into() });
    }

    #[test]
    fn a_correct_partial_warp_mask_is_predicted_to_be_reported_but_never_gated() {
        // Read literally, the documentation predicts a gating RC002 here: the
        // call site is not provably convergent, and MUTATION.md says v1 does no
        // mask arithmetic. That would make this valid program a false positive
        // at a gating tier. Measurement says reconverge reports it at warning
        // tier and lets it through -- so the rule follows the measurement, and
        // the precision requirement survives a case built to break it.
        let k = Kernel::new(vec![Stmt::If {
            pred: even(),
            body: vec![Stmt::Ballot { mask: Mask::Literal(0x5555_5555) }],
        }]);
        let sem = interpret(&k, Launch::one_block(32));
        assert_eq!(sem.oracle, crate::oracle::ConstructionOracle::KnownMaskValid);
        let p = predict(&sem);
        assert_eq!(p.expected, ExpectedStatic::WarningOnly { code: "RC002".into() });
        assert!(matches!(p.provenance, Provenance::Measured { .. }));
    }

    #[test]
    fn a_mask_naming_an_absent_lane_is_predicted_to_gate() {
        // The other side of the same rule: the full mask under the same guard
        // names sixteen lanes that never arrive, and that is what gates.
        let k = Kernel::new(vec![Stmt::If {
            pred: even(),
            body: vec![Stmt::Ballot { mask: Mask::Literal(0xffff_ffff) }],
        }]);
        let p = predict(&interpret(&k, Launch::one_block(32)));
        assert_eq!(p.expected, ExpectedStatic::Gating { code: "RC002".into() });
    }

    #[test]
    fn a_loop_inside_the_guard_is_replayed_but_a_guard_inside_the_loop_is_not() {
        // The minimized pair. Same two constructs, nested the other way round,
        // and only one of them gates. Neither shape appears in any
        // documentation, and the loop's trip count is uniform in both.
        let uniform_bound = Value::Const(2);
        let guard_inside_loop = Kernel::new(vec![Stmt::Loop {
            bound: uniform_bound.clone(),
            body: vec![Stmt::If { pred: even(), body: vec![Stmt::Barrier] }],
        }]);
        let loop_inside_guard = Kernel::new(vec![Stmt::If {
            pred: even(),
            body: vec![Stmt::Loop { bound: uniform_bound, body: vec![Stmt::Barrier] }],
        }]);

        let a = predict_kernel(&guard_inside_loop);
        assert_eq!(a.expected, ExpectedStatic::WarningOnly { code: "RC001".into() });
        assert_eq!(a.rule, "barrier_under_guard_inside_loop");
        assert!(matches!(a.provenance, Provenance::Measured { .. }));

        let b = predict_kernel(&loop_inside_guard);
        assert_eq!(b.expected, ExpectedStatic::Gating { code: "RC001".into() });

        // Both programs are equally undefined; only the analyzer's confidence
        // in saying so differs.
        for k in [&guard_inside_loop, &loop_inside_guard] {
            assert_eq!(
                interpret(k, Launch::one_block(32)).oracle,
                crate::oracle::ConstructionOracle::KnownUnsafe
            );
        }
    }

    #[test]
    fn only_findings_sharing_the_witnessed_divergence_source_are_gated() {
        let lane_env = || Pred::Cmp(Value::WarpId, CmpOp::Eq, 0);

        // Two barriers, two different sources. The first is attempted and gates;
        // the second diverges for a different reason and only warns.
        let two_sources = Kernel::new(vec![
            Stmt::If { pred: even(), body: vec![Stmt::Barrier] },
            Stmt::If { pred: Pred::Not(Box::new(even())), body: vec![Stmt::Barrier] },
        ]);
        assert_eq!(
            predict_kernel(&two_sources).expected,
            ExpectedStatic::Gating { code: "RC001".into() },
            "the first source is witnessed, so the case gates"
        );

        // Same two barriers, one source. Measured: both confirmed, two witnesses.
        let one_source = Kernel::new(vec![
            Stmt::If { pred: even(), body: vec![Stmt::Barrier] },
            Stmt::If { pred: even(), body: vec![Stmt::Barrier] },
        ]);
        assert_eq!(
            predict_kernel(&one_source).expected,
            ExpectedStatic::Gating { code: "RC001".into() }
        );

        // Lane-environment source FIRST: nothing in the function is promotable,
        // so a confirmable barrier below it leaves the CI gate.
        let blocked = Kernel::new(vec![
            Stmt::If { pred: lane_env(), body: vec![Stmt::Barrier] },
            Stmt::If { pred: even(), body: vec![Stmt::Barrier] },
        ]);
        let p = predict_kernel(&blocked);
        assert_eq!(p.expected, ExpectedStatic::WarningOnly { code: "RC001".into() });
        assert!(matches!(p.provenance, Provenance::Measured { .. }));

        // The same source SECOND: measured to change nothing.
        let unaffected = Kernel::new(vec![
            Stmt::If { pred: even(), body: vec![Stmt::Barrier] },
            Stmt::If { pred: lane_env(), body: vec![Stmt::Barrier] },
        ]);
        assert_eq!(
            predict_kernel(&unaffected).expected,
            ExpectedStatic::Gating { code: "RC001".into() },
            "appending a lane-environment barrier must not un-gate the one above it"
        );
    }

    #[test]
    fn the_held_out_case_that_refuted_the_previous_version_of_the_rule() {
        // Two collectives under complementary guards, both with mask 0x1. The
        // first names only a lane that is present; the second names lane 0, which
        // is absent there. The previous rule predicted gating on the second;
        // measurement said warning on both, because the source that got the
        // witness was the first one and nothing was wrong under it.
        let k = Kernel::new(vec![
            Stmt::If {
                pred: even(),
                body: vec![Stmt::Ballot { mask: Mask::Literal(0x0000_0001) }],
            },
            Stmt::If {
                pred: Pred::Not(Box::new(even())),
                body: vec![Stmt::Ballot { mask: Mask::Literal(0x0000_0001) }],
            },
        ]);
        let sem = interpret(&k, Launch::one_block(32));
        assert_eq!(sem.oracle, crate::oracle::ConstructionOracle::KnownMaskInvalid);
        let p = predict(&sem);
        assert_eq!(p.expected, ExpectedStatic::WarningOnly { code: "RC002".into() });
        assert_eq!(p.rule, "another_divergence_source_got_the_witness");
    }
}
