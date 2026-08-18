//! The engine is held to the measurements.
//!
//! Stage 3 ran fourteen hand-written templates against reconverge 0.1.6 on a
//! real toolchain and recorded fourteen rows (`docs/conformance-reconverge-0.1.6.md`).
//! Stage 4 replaces the hand-declared labels with two computations: the oracle
//! from executing the kernel, and the prediction from a rule table.
//!
//! Nothing licenses trusting those computations on generated kernels except
//! this: they must reproduce the fourteen rows that were measured. So every
//! seed is checked three ways -- it must render the *same program* as the hand
//! template, compute the *same oracle*, and derive the *same prediction*.
//!
//! One seed deliberately does not match, and that is recorded here rather than
//! smoothed over: see `warp_id_is_the_one_seed_whose_oracle_is_launch_dependent`.

use simt_diff::interpret::interpret;
use simt_diff::model::predict;
use simt_diff::mutate;
use simt_diff::oracle::ConstructionOracle;
use simt_diff::prediction::ExpectedStatic;
use simt_diff::records::Launch;
use simt_diff::templates;

/// The launch every Stage 3 row was measured at.
fn launch() -> Launch {
    Launch::one_block(32)
}

#[test]
fn every_hand_written_template_has_an_ir_seed() {
    let seeds: Vec<&str> = mutate::seeds().into_iter().map(|(n, _)| n).collect();
    for t in templates::TEMPLATES {
        assert!(
            seeds.contains(&t.id),
            "{} has no IR seed, so the mutation engine cannot reach anything \
             derived from it",
            t.id
        );
    }
    assert_eq!(seeds.len(), templates::TEMPLATES.len());
}

#[test]
fn each_seed_renders_the_same_program_as_its_hand_written_template() {
    for (name, kernel) in mutate::seeds() {
        let hand = templates::find(name).expect("seed names a template");
        assert_eq!(
            kernel.render_body(),
            hand.body,
            "{name}: the IR renders a different body than the template that was \
             measured, so the seed is not the program Stage 3 ran"
        );
        assert_eq!(
            kernel.extra_items(),
            hand.extra_items,
            "{name}: items differ"
        );
        assert_eq!(
            kernel.extra_uses(),
            hand.extra_uses.to_vec(),
            "{name}: uses differ"
        );
    }
}

#[test]
fn each_seed_computes_the_oracle_its_template_declared() {
    // The exception is stated as data, not hidden in a filter: at block=32 a
    // whole-warp guard is uniform, and the interpreter is right about that.
    let launch_dependent = ["barrier_guarded_by_warp_id"];

    for (name, kernel) in mutate::seeds() {
        if launch_dependent.contains(&name) {
            continue;
        }
        let hand = templates::find(name).unwrap();
        let computed = interpret(&kernel, launch()).oracle;
        assert_eq!(
            computed, hand.oracle,
            "{name}: executing the kernel says {computed:?} while the template \
             declared {:?}",
            hand.oracle
        );
    }
}

#[test]
fn each_seed_derives_the_prediction_its_template_declared() {
    for (name, kernel) in mutate::seeds() {
        let hand = templates::find(name).unwrap();
        let expected: ExpectedStatic = hand.expected_static.into();
        let derived = predict(&interpret(&kernel, launch()));
        assert_eq!(
            derived.expected, expected,
            "{name}: the rule table derives {:?} but the template declared \
             {expected:?} and the measured row agreed with the template -- the \
             rule table is wrong, not the measurement (rule fired: {})",
            derived.expected, derived.rule
        );
    }
}

#[test]
fn the_reference_models_agree_wherever_both_exist() {
    for (name, kernel) in mutate::seeds() {
        let hand = templates::find(name).unwrap();
        let computed = interpret(&kernel, launch()).reference;
        let declared = (hand.reference)(launch());
        match (computed, declared) {
            (Some(c), Some(d)) => assert_eq!(
                c.expected, d.expected,
                "{name}: computed and hand-written reference models disagree"
            ),
            (None, None) => {}
            (c, d) => {
                // Only the launch-dependent seed may differ here, for the same
                // reason its oracle does.
                assert_eq!(
                    name,
                    "barrier_guarded_by_warp_id",
                    "{name}: one side has a reference model and the other does \
                     not (computed: {}, declared: {})",
                    c.is_some(),
                    d.is_some()
                );
            }
        }
    }
}

#[test]
fn warp_id_is_the_one_seed_whose_oracle_is_launch_dependent() {
    // The hand-written template declares KNOWN_UNSAFE and justifies it with
    // "warp_id() is uniform within a warp but differs across warps, so whole
    // warps skip the block barrier". That reasoning needs more than one warp,
    // and every Stage 3 row was measured at block=32 -- where the guard is
    // true for every thread and nothing diverges at all.
    //
    // The analyzer's behaviour is unaffected: a lane-environment guard is a
    // divergence source it cannot evaluate, so RC001 at warning tier is
    // correct at either launch, and that is what was measured. What changes is
    // the construction label, which is now computed per launch instead of
    // asserted once.
    let kernel = mutate::seed("barrier_guarded_by_warp_id").unwrap();
    let hand = templates::find("barrier_guarded_by_warp_id").unwrap();
    assert_eq!(hand.oracle, ConstructionOracle::KnownUnsafe);

    let at_32 = interpret(&kernel, Launch::one_block(32));
    assert_eq!(at_32.oracle, ConstructionOracle::KnownSafe);
    assert!(
        at_32.reference.is_some(),
        "a safe kernel has defined output"
    );

    let at_64 = interpret(&kernel, Launch::one_block(64));
    assert_eq!(
        at_64.oracle, hand.oracle,
        "the template's label holds from two warps up"
    );

    // The prediction is the same either way, which is why the Stage 3 row held.
    let expected: ExpectedStatic = hand.expected_static.into();
    assert_eq!(predict(&at_32).expected, expected);
    assert_eq!(predict(&at_64).expected, expected);
}

#[test]
fn the_seeds_reproduce_the_measured_conformance_table() {
    // The fourteen rows of docs/conformance-reconverge-0.1.6.md, as data. If a
    // rule-table change breaks one of these, it broke a measurement.
    let measured: &[(&str, &str)] = &[
        ("mask_full_convergent", "silent"),
        ("mask_shrunk_convergent", "silent"),
        ("barrier_uniform", "silent"),
        ("barrier_divergent_intra_warp", "RC001 gating"),
        ("barrier_divergent_nested", "RC001 gating"),
        ("barrier_divergent_loop_break", "RC001 gating"),
        ("barrier_in_helper_divergent_call", "RC001 warning-only"),
        ("barrier_in_helper_uniform_call", "silent"),
        ("barrier_guarded_by_lanemask", "RC001 warning-only"),
        ("barrier_guarded_by_warp_id", "RC001 warning-only"),
        ("collective_under_divergence", "RC002 gating"),
        ("mask_from_named_const", "silent"),
        ("mask_from_active_mask", "silent"),
        ("collective_unmasked_wrapper", "silent"),
    ];
    assert_eq!(measured.len(), templates::TEMPLATES.len());

    for (name, row) in measured {
        let kernel = mutate::seed(name).unwrap_or_else(|| panic!("{name} has no seed"));
        let derived = predict(&interpret(&kernel, launch()));
        let rendered = match &derived.expected {
            ExpectedStatic::Gating { code } => format!("{code} gating"),
            ExpectedStatic::WarningOnly { code } => format!("{code} warning-only"),
            ExpectedStatic::Silent => "silent".to_string(),
            ExpectedStatic::Unspecified => "-".to_string(),
        };
        assert_eq!(
            &rendered, row,
            "{name}: the model no longer predicts the measured row"
        );
    }
}

#[test]
fn the_engine_reaches_cases_the_hand_written_corpus_never_had() {
    let all = mutate::enumerate(1, launch());
    let ids: Vec<String> = all.iter().map(|m| m.id.clone()).collect();

    // Each of these crosses a documented boundary that no Stage 3 template did.
    for wanted in [
        // the same divergence behind a source the analyzer cannot evaluate
        "barrier_divergent_intra_warp+to_lanemask@0",
        // a guard that needs a truncating cast to replay
        "barrier_divergent_intra_warp+truncate_operand@0",
        // an evaluable predicate conjoined with an unevaluable one
        "barrier_divergent_intra_warp+conjoin_lane_env@0",
        // two calls between the divergence and the barrier
        "barrier_in_helper_divergent_call+deepen_helper",
        // valid by construction at a call site that cannot be proven convergent
        "collective_under_divergence+mask_match_participants@0.0",
    ] {
        assert!(
            ids.contains(&wanted.to_string()),
            "the engine never generates {wanted}"
        );
    }

    // Coverage is asserted over programs, not lineages: two operator chains that
    // render the same kernel are one case, and which lineage survives dedup is
    // an artefact of enumeration order. `if i.get() % 4 == 0` is reachable both
    // by widening the intra-warp guard's modulus and by nesting a guard around
    // the uniform barrier; only the first-enumerated name survives.
    for construct in [
        "(i.get() as u8) as u32 % 2 == 0",
        "warp::lanemask_lt().count_ones() % 2 == 0",
        "i.get() % 2 == 0 && warp::warp_id() == 0",
        "i.get() % 4 == 0 {\n            thread::sync_threads();",
        "0x5555_5555",
    ] {
        assert!(
            all.iter()
                .any(|m| m.kernel.render_body().contains(construct)),
            "no generated kernel contains `{construct}`"
        );
    }
    assert!(
        all.iter().any(|m| m.kernel.helper_depth == 2),
        "the engine never reaches a two-call chain"
    );

    let unsafe_count = all
        .iter()
        .filter(|m| interpret(&m.kernel, launch()).oracle == ConstructionOracle::KnownUnsafe)
        .count();
    let safe_count = all
        .iter()
        .filter(|m| interpret(&m.kernel, launch()).oracle.asserts_valid())
        .count();
    // A corpus that is all bugs cannot measure precision, and one that is all
    // clean cannot measure recall.
    assert!(unsafe_count >= 10, "only {unsafe_count} unsafe cases");
    assert!(safe_count >= 10, "only {safe_count} valid cases");
}
