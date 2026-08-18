//! The deterministic mutation engine (brief §11).
//!
//! Fourteen hand-written templates produced fourteen held predictions. That
//! result is real, and it is also the limit of what hand-writing can reach: it
//! says nothing about constructs nobody thought to write. This module writes
//! them, by transforming the [`crate::ir`] forms of those fourteen seeds.
//!
//! Three properties are what make the output worth analyzing:
//!
//! 1. **No inherited labels.** Every mutant's oracle, reference model and
//!    prediction are recomputed from the mutant by [`crate::interpret`] and
//!    [`crate::model`]. A mutation that turns an unsafe kernel safe is
//!    relabelled, not mislabelled -- which is the brief's §11 requirement and
//!    the reason reconverge's own `delbar` row exists.
//! 2. **Determinism.** Enumeration order is structural, identity is the SHA-256
//!    of the rendered program, and sampling uses a named seed. The same command
//!    on another machine produces the same corpus.
//! 3. **Targeting.** The operators are not random edits. Each one moves a
//!    kernel across a boundary the analyzer's documentation draws: into a call,
//!    onto a lane-environment register, behind a truncating cast, or onto a
//!    mask the analyzer cannot evaluate.

use std::collections::BTreeSet;

use crate::interpret::{Semantics, interpret};
use crate::ir::{CmpOp, Kernel, Mask, Pred, Stmt, Value};
use crate::model::{ModelPrediction, predict};
use crate::records::{GeneratorRecord, Launch, PredictionBasis};
use crate::templates::{GENERATOR_VERSION, render_kernel_file};

/// One generated kernel with its ancestry.
#[derive(Clone, Debug)]
pub struct Mutant {
    /// `<seed>+<op>@<address>[+<op>@<address>]`, or the bare seed name at
    /// depth 0. Stable across machines.
    pub id: String,
    pub seed: String,
    pub lineage: Vec<String>,
    pub kernel: Kernel,
}

impl Mutant {
    pub fn depth(&self) -> usize {
        self.lineage.len()
    }
}

fn even() -> Pred {
    Pred::Cmp(Value::Rem(Box::new(Value::LaneIndex), 2), CmpOp::Eq, 0)
}

fn quarter() -> Pred {
    Pred::Cmp(Value::Rem(Box::new(Value::LaneIndex), 4), CmpOp::Eq, 0)
}

/// The Stage 3 templates, in IR. `tests/ir_seeds.rs` holds these to the
/// measured conformance rows, which is what licenses trusting the engine's
/// verdict on kernels nobody has measured.
pub fn seeds() -> Vec<(&'static str, Kernel)> {
    vec![
        ("mask_full_convergent", Kernel::new(vec![Stmt::Ballot { mask: Mask::Literal(0xffff_ffff) }])),
        ("mask_shrunk_convergent", Kernel::new(vec![Stmt::Ballot { mask: Mask::Literal(0x0000_ffff) }])),
        ("barrier_uniform", Kernel::new(vec![Stmt::Barrier])),
        (
            "barrier_divergent_intra_warp",
            Kernel::new(vec![Stmt::If { pred: even(), body: vec![Stmt::Barrier] }]),
        ),
        (
            "barrier_divergent_nested",
            Kernel::new(vec![Stmt::If {
                pred: even(),
                body: vec![Stmt::If { pred: quarter(), body: vec![Stmt::Barrier] }],
            }]),
        ),
        (
            "barrier_divergent_loop_break",
            Kernel::new(vec![Stmt::Loop {
                bound: Value::Rem(Box::new(Value::LaneIndex), 4),
                body: vec![Stmt::Barrier],
            }]),
        ),
        (
            "barrier_in_helper_divergent_call",
            Kernel::with_helper(vec![Stmt::If { pred: even(), body: vec![Stmt::CallHelper] }], 1),
        ),
        ("barrier_in_helper_uniform_call", Kernel::with_helper(vec![Stmt::CallHelper], 1)),
        (
            "barrier_guarded_by_lanemask",
            Kernel::new(vec![Stmt::If {
                pred: Pred::Cmp(Value::LaneMaskLtPopcount, CmpOp::Gt, 4),
                body: vec![Stmt::Barrier],
            }]),
        ),
        (
            "barrier_guarded_by_warp_id",
            Kernel::new(vec![Stmt::If {
                pred: Pred::Cmp(Value::WarpId, CmpOp::Eq, 0),
                body: vec![Stmt::Barrier],
            }]),
        ),
        (
            "collective_under_divergence",
            Kernel::new(vec![Stmt::If {
                pred: even(),
                body: vec![Stmt::Ballot { mask: Mask::Literal(0xffff_ffff) }],
            }]),
        ),
        (
            "mask_from_named_const",
            Kernel::new(vec![Stmt::Ballot { mask: Mask::NamedConst(0xffff_ffff) }]),
        ),
        ("mask_from_active_mask", Kernel::new(vec![Stmt::Ballot { mask: Mask::ActiveMask }])),
        (
            "collective_unmasked_wrapper",
            Kernel::new(vec![Stmt::If {
                pred: even(),
                body: vec![Stmt::Ballot { mask: Mask::ImplicitWrapper }],
            }]),
        ),
    ]
}

pub fn seed(name: &str) -> Option<Kernel> {
    seeds().into_iter().find(|(n, _)| *n == name).map(|(_, k)| k)
}

// --------------------------------------------------------------- operators ---

/// Rewrite the first value in a predicate, in pre-order. Returns whether the
/// rewrite applied, so an operator that cannot fire produces no mutant rather
/// than a duplicate.
fn map_first_value(p: &mut Pred, f: &mut dyn FnMut(&mut Value) -> bool) -> bool {
    match p {
        Pred::Cmp(v, _, _) => f(v),
        Pred::Not(inner) => map_first_value(inner, f),
        Pred::And(a, b) => map_first_value(a, f) || map_first_value(b, f),
    }
}

/// Replace the leaf of an arithmetic chain, keeping the arithmetic. Turning
/// `i.get() % 2 == 0` into `warp_id() % 2 == 0` keeps the shape of the guard
/// and changes only whether the analyzer can evaluate its source.
fn replace_leaf(v: &mut Value, leaf: Value) {
    match v {
        Value::LaneIndex | Value::WarpId | Value::LaneMaskLtPopcount | Value::Const(_) => {
            *v = leaf
        }
        Value::Rem(inner, _)
        | Value::Div(inner, _)
        | Value::BitAnd(inner, _)
        | Value::TruncU8(inner) => replace_leaf(inner, leaf),
    }
}

fn wrap_leaf_in_cast(v: &mut Value) -> bool {
    if v.has_truncating_cast() {
        return false;
    }
    fn go(v: &mut Value) {
        match v {
            Value::LaneIndex | Value::WarpId | Value::LaneMaskLtPopcount | Value::Const(_) => {
                *v = Value::TruncU8(Box::new(v.clone()));
            }
            Value::Rem(inner, _)
            | Value::Div(inner, _)
            | Value::BitAnd(inner, _)
            | Value::TruncU8(inner) => go(inner),
        }
    }
    go(v);
    true
}

fn double_modulus(v: &mut Value) -> bool {
    match v {
        Value::Rem(_, k) | Value::Div(_, k) if *k * 2 <= 32 => {
            *k *= 2;
            true
        }
        _ => false,
    }
}

/// Every single-step mutation of `kernel`, in a fixed order.
///
/// `launch` matters: `mask_match_participants` needs to know which lanes
/// actually reach the collective, so the operator set is a function of the
/// launch the corpus is being built for.
pub fn mutations(kernel: &Kernel, launch: Launch) -> Vec<(String, Kernel)> {
    let sem = interpret(kernel, launch);
    let mut out: Vec<(String, Kernel)> = Vec::new();
    let sites: Vec<(Vec<usize>, Stmt)> =
        kernel.walk().into_iter().map(|(p, s)| (p, s.clone())).collect();

    let mut push = |name: String, k: Kernel| {
        if &k != kernel {
            out.push((name, k));
        }
    };

    for (path, stmt) in &sites {
        let tag = path.iter().map(usize::to_string).collect::<Vec<_>>().join(".");

        // ---- guard rewrites -------------------------------------------------
        if let Stmt::If { pred, body } = stmt {
            for (op, transform) in guard_rewrites() {
                let mut k = kernel.clone();
                let mut fired = false;
                if let Some((vec, idx)) = k.locate_mut(path)
                    && let Stmt::If { pred, .. } = &mut vec[idx]
                {
                    fired = transform(pred);
                }
                if fired {
                    push(format!("{op}@{tag}"), k);
                }
            }

            // The guard disappears: the barrier becomes uniform and the oracle
            // must flip. reconverge's own corpus keeps this class (`delbar`) as
            // a control, and so does this.
            let mut dropped = kernel.clone();
            if let Some((vec, idx)) = dropped.locate_mut(path) {
                let old = vec.remove(idx);
                if let Stmt::If { body, .. } = old {
                    for (j, s) in body.into_iter().enumerate() {
                        vec.insert(idx + j, s);
                    }
                }
            }
            push(format!("drop_guard@{tag}"), dropped);

            // A complementary sibling: every thread now arrives exactly once,
            // at one of two program points. Path-insensitive analysis and the
            // programming model disagree about this one, which is why it is
            // here.
            let mut paired = kernel.clone();
            if let Some((vec, idx)) = paired.locate_mut(path) {
                vec.insert(
                    idx + 1,
                    Stmt::If { pred: Pred::Not(Box::new(pred.clone())), body: body.clone() },
                );
            }
            push(format!("complementary_guard@{tag}"), paired);
        }

        // ---- loop bound rewrites -------------------------------------------
        if matches!(stmt, Stmt::Loop { .. }) {
            for (op, transform) in value_rewrites() {
                let mut k = kernel.clone();
                let mut fired = false;
                if let Some((vec, idx)) = k.locate_mut(path)
                    && let Stmt::Loop { bound, .. } = &mut vec[idx]
                {
                    fired = transform(bound);
                }
                if fired {
                    push(format!("{op}_bound@{tag}"), k);
                }
            }
        }

        // ---- make a loop's trip count uniform ------------------------------
        // Every other bound here is thread-derived, so without this the corpus
        // cannot express "a divergent guard inside a loop that is itself
        // uniform" -- and that is the one shape that separates the loop from
        // the guard as the reason a finding was not promoted.
        if let Stmt::Loop { bound, .. } = stmt
            && !matches!(bound, Value::Const(_))
        {
            let mut k = kernel.clone();
            if let Some((vec, idx)) = k.locate_mut(path)
                && let Stmt::Loop { bound, .. } = &mut vec[idx]
            {
                *bound = Value::Const(2);
            }
            push(format!("uniform_bound@{tag}"), k);
        }

        // ---- structural rewrites, applicable to any statement ---------------
        for (op, wrapper) in [
            ("nest_guard", Wrapper::Guard),
            ("wrap_in_loop", Wrapper::Loop),
        ] {
            let mut k = kernel.clone();
            if let Some((vec, idx)) = k.locate_mut(path) {
                let old = vec[idx].clone();
                vec[idx] = match wrapper {
                    Wrapper::Guard => Stmt::If { pred: quarter(), body: vec![old] },
                    Wrapper::Loop => Stmt::Loop {
                        bound: Value::Rem(Box::new(Value::LaneIndex), 4),
                        body: vec![old],
                    },
                };
            }
            push(format!("{op}@{tag}"), k);
        }

        // ---- an unrelated unevaluable barrier beside this one --------------
        // Two operators, differing only in where the new barrier goes, because
        // that turned out to be the whole point: a lane-environment barrier
        // *above* a confirmable one takes it out of the gate and the same
        // barrier *below* it does not. The original guard is untouched in both,
        // so any change in the verdict can only come from the addition.
        if matches!(stmt, Stmt::Barrier | Stmt::CallHelper) && path.len() > 1 {
            for (op, after) in [("add_lane_env_sibling", true), ("prepend_lane_env_sibling", false)]
            {
                let mut k = kernel.clone();
                let outer = &path[..path.len() - 1];
                if let Some((vec, idx)) = k.locate_mut(outer) {
                    let at = if after { idx + 1 } else { idx };
                    vec.insert(
                        at,
                        Stmt::If {
                            pred: Pred::Cmp(Value::WarpId, CmpOp::Eq, 0),
                            body: vec![Stmt::Barrier],
                        },
                    );
                }
                push(format!("{op}@{tag}"), k);
            }
        }

        // ---- the same guard again, at the end of the function --------------
        // The discriminator for "which findings share the witnessed source". Two
        // sites under one guard are both confirmed and two under different
        // guards are not, but nothing yet says whether a *later* site sharing
        // the first one's guard is promoted when a different guard sits between
        // them. This operator builds exactly that: A, B, A.
        if matches!(stmt, Stmt::Barrier | Stmt::CallHelper) && path.len() == 2 {
            let mut k = kernel.clone();
            if let Some(Stmt::If { .. }) = k.get(&path[..1]).cloned()
                && let Some(clone) = k.get(&path[..1]).cloned()
            {
                k.stmts.push(clone);
                push(format!("clone_guard_to_end@{tag}"), k);
            }
        }

        // ---- put the barrier behind a call ---------------------------------
        if matches!(stmt, Stmt::Barrier) {
            let mut k = kernel.clone();
            if let Some((vec, idx)) = k.locate_mut(path) {
                vec[idx] = Stmt::CallHelper;
            }
            k.helper_depth = k.helper_depth.max(1);
            push(format!("hoist_into_helper@{tag}"), k);
        }

        // ---- mask rewrites --------------------------------------------------
        if let Stmt::Ballot { mask } = stmt {
            let participants = sem
                .sites
                .iter()
                .find(|s| s.key == format!("collective@{tag}"))
                .and_then(|s| s.participants.values().copied().next());
            let mut candidates: Vec<(String, Mask)> = vec![
                ("mask_widen".to_string(), Mask::Literal(0xffff_ffff)),
                ("mask_shrink".to_string(), Mask::Literal(0x0000_ffff)),
                ("mask_single_lane".to_string(), Mask::Literal(0x0000_0001)),
                ("mask_to_named_const".to_string(), Mask::NamedConst(0xffff_ffff)),
                ("mask_to_active_mask".to_string(), Mask::ActiveMask),
                ("mask_to_wrapper".to_string(), Mask::ImplicitWrapper),
            ];
            // The correct guarded partial-warp idiom, spelled as a literal:
            // valid by construction at a call site the analyzer cannot prove
            // convergent. If anything in this corpus is a false positive at a
            // gating tier, it is this.
            if let Some(p) = participants {
                candidates.push(("mask_match_participants".to_string(), Mask::Literal(p)));
            }
            for (op, new_mask) in candidates {
                if new_mask == *mask {
                    continue;
                }
                let mut k = kernel.clone();
                if let Some((vec, idx)) = k.locate_mut(path)
                    && let Stmt::Ballot { mask } = &mut vec[idx]
                {
                    *mask = new_mask;
                }
                push(format!("{op}@{tag}"), k);
            }
        }
    }

    // ---- whole-kernel rewrites ---------------------------------------------
    if kernel.helper_depth == 1 {
        let mut k = kernel.clone();
        k.helper_depth = 2;
        push("deepen_helper".to_string(), k);
    }

    out
}

enum Wrapper {
    Guard,
    Loop,
}

type GuardRewrite = (&'static str, Box<dyn Fn(&mut Pred) -> bool>);

fn guard_rewrites() -> Vec<GuardRewrite> {
    let mut v: Vec<GuardRewrite> = vec![
        (
            "invert_guard",
            Box::new(|p: &mut Pred| {
                *p = Pred::Not(Box::new(p.clone()));
                true
            }),
        ),
        (
            "negate_cmp",
            Box::new(|p: &mut Pred| match p {
                // The same condition as `invert_guard`, spelled without the
                // `!`. If the two ever classify differently the difference is
                // syntactic, and that is worth knowing.
                Pred::Cmp(_, op, _) => {
                    *op = op.negate();
                    true
                }
                _ => false,
            }),
        ),
        (
            "shift_bound",
            Box::new(|p: &mut Pred| match p {
                Pred::Cmp(_, _, rhs) => {
                    *rhs += 1;
                    true
                }
                _ => false,
            }),
        ),
        (
            "conjoin_lane_env",
            Box::new(|p: &mut Pred| {
                if p.reads_lane_environment() {
                    return false;
                }
                *p = Pred::And(
                    Box::new(p.clone()),
                    Box::new(Pred::Cmp(Value::WarpId, CmpOp::Eq, 0)),
                );
                true
            }),
        ),
    ];
    for (name, f) in value_rewrites() {
        v.push((name, Box::new(move |p: &mut Pred| map_first_value(p, &mut |val| f(val)))));
    }
    v
}

type ValueRewrite = (&'static str, std::rc::Rc<dyn Fn(&mut Value) -> bool>);

fn value_rewrites() -> Vec<ValueRewrite> {
    use std::rc::Rc;
    vec![
        ("retarget_modulus", Rc::new(double_modulus) as Rc<dyn Fn(&mut Value) -> bool>),
        ("truncate_operand", Rc::new(wrap_leaf_in_cast)),
        (
            "to_warp_id",
            Rc::new(|v: &mut Value| {
                if !v.reads_lane_index() {
                    return false;
                }
                replace_leaf(v, Value::WarpId);
                true
            }),
        ),
        (
            "to_lanemask",
            Rc::new(|v: &mut Value| {
                if !v.reads_lane_index() {
                    return false;
                }
                replace_leaf(v, Value::LaneMaskLtPopcount);
                true
            }),
        ),
    ]
}

// -------------------------------------------------------------- enumeration ---

/// Every mutant reachable from the seeds in at most `depth` steps, deduplicated
/// by the program it renders to.
///
/// Depth 0 is the seeds themselves: the corpus always contains its own
/// controls.
pub fn enumerate(depth: usize, launch: Launch) -> Vec<Mutant> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut all: Vec<Mutant> = Vec::new();
    let mut frontier: Vec<Mutant> = Vec::new();

    for (name, kernel) in seeds() {
        let m = Mutant {
            id: name.to_string(),
            seed: name.to_string(),
            lineage: Vec::new(),
            kernel,
        };
        if seen.insert(fingerprint(&m.kernel)) {
            all.push(m.clone());
            frontier.push(m);
        }
    }

    for _ in 0..depth {
        let mut next = Vec::new();
        for parent in &frontier {
            for (op, kernel) in mutations(&parent.kernel, launch) {
                if !seen.insert(fingerprint(&kernel)) {
                    continue;
                }
                let mut lineage = parent.lineage.clone();
                lineage.push(op.clone());
                let m = Mutant {
                    id: format!("{}+{}", parent.id, op),
                    seed: parent.seed.clone(),
                    lineage,
                    kernel,
                };
                all.push(m.clone());
                next.push(m);
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }

    all
}

/// The rendered program, which is the only thing that decides whether two
/// mutants are the same case. Two different operator chains reaching the same
/// text are one case.
pub fn fingerprint(kernel: &Kernel) -> String {
    format!(
        "{}|{}|{}",
        kernel.helper_depth,
        kernel.extra_items(),
        kernel.render_body()
    )
}

/// A reproducible subset, when the whole space is more than a sweep can afford.
///
/// Taking the first `limit` in enumeration order would sample only the earliest
/// operators on the earliest seeds; a seeded shuffle keeps the subset spread
/// across the space and still identical on every machine.
pub fn sample(mut mutants: Vec<Mutant>, limit: usize, seed: u64) -> Vec<Mutant> {
    if mutants.len() <= limit {
        return mutants;
    }
    // Depth 0 is never dropped: the seeds are the controls that make the rest
    // of the table readable.
    let controls: Vec<Mutant> = mutants.iter().filter(|m| m.depth() == 0).cloned().collect();
    mutants.retain(|m| m.depth() > 0);
    mutants.sort_by(|a, b| a.id.cmp(&b.id));

    // The constant is arbitrary; it exists only so seed 0 does not start the
    // generator at its own fixed point.
    let mut state = seed ^ 0x5347_4d54_5f44_4946;
    let mut i = mutants.len();
    while i > 1 {
        let j = (splitmix64(&mut state) % i as u64) as usize;
        i -= 1;
        mutants.swap(i, j);
    }
    let room = limit.saturating_sub(controls.len());
    let mut out = controls;
    out.extend(mutants.into_iter().take(room));
    out.sort_by_key(|m| (m.depth(), m.id.clone()));
    out
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ------------------------------------------------------------------ records ---

/// Build the case record for a mutant: oracle and reference model computed by
/// the interpreter, prediction computed by the documented model, both attributed
/// in the generated source so a human reading the case can see where each claim
/// came from.
pub fn record(m: &Mutant, seed: u64, launches: Vec<Launch>) -> GeneratorRecord {
    let primary = launches.first().copied().unwrap_or(Launch::one_block(32));
    let sem = interpret(&m.kernel, primary);
    let prediction = predict(&sem);
    let source = kernel_source(m, &sem, &prediction, primary);

    GeneratorRecord {
        template_id: m.id.clone(),
        generator_version: GENERATOR_VERSION.to_string(),
        seed,
        oracle: sem.oracle,
        oracle_reason: sem.oracle_reason.clone(),
        kernel_sha256: crate::templates::sha256_hex_public(source.as_bytes()),
        kernel_source: source,
        kernel_name: "probe".to_string(),
        launches,
        reference_model: sem.reference.clone(),
        documented_limitation: prediction.documented_limitation.clone(),
        expected_static: prediction.expected.clone(),
        prediction_basis: Some(PredictionBasis {
            rule: prediction.rule.clone(),
            provenance: prediction.provenance.clone(),
            mutation_lineage: m.lineage.clone(),
            seed_template: m.seed.clone(),
        }),
    }
}

/// A record for a kernel that is not a corpus mutant -- a minimizer candidate,
/// or anything else hand-built. The oracle and the prediction are computed the
/// same way, so such a case is not a second-class citizen.
pub fn record_for_kernel(kernel: &Kernel, id: &str, launch: Launch) -> GeneratorRecord {
    let m = Mutant {
        id: id.to_string(),
        seed: id.to_string(),
        lineage: Vec::new(),
        kernel: kernel.clone(),
    };
    record(&m, 0, vec![launch])
}

/// Rebuild the record for a case whose recipe is known.
///
/// The lineage is not decoration: it is written into the generated kernel's doc
/// comment, so a record rebuilt without it hashes differently. Comparing that
/// hash against a stored one is how generator drift is detected, and getting
/// this wrong made every rebuilt case look like drift.
pub fn record_from_recipe(
    template_id: &str,
    seed_template: &str,
    lineage: &[String],
    kernel: &Kernel,
    launch: Launch,
) -> GeneratorRecord {
    let m = Mutant {
        id: template_id.to_string(),
        seed: seed_template.to_string(),
        lineage: lineage.to_vec(),
        kernel: kernel.clone(),
    };
    record(&m, 0, vec![launch])
}

fn kernel_source(
    m: &Mutant,
    sem: &Semantics,
    prediction: &ModelPrediction,
    launch: Launch,
) -> String {
    let lineage = if m.lineage.is_empty() {
        "none (seed template)".to_string()
    } else {
        m.lineage.join(" then ")
    };
    // The oracle and the mutation are properties of the program, so they belong
    // in it. The prediction is a property of the *model*, and putting it here
    // would mean every rule change rewrote every kernel and renamed every case
    // -- a regression corpus cannot have identities that move like that. It
    // lives in generator.json, which is where a reader is pointed.
    let doc = format!(
        "ORACLE: {:?}, computed by executing every thread of block={} -- {}\n\
         //!\n\
         //! MUTATION: {} of {}\n\
         //!\n\
         //! The prediction for this case, and what it rests on, are in \
         generator.json.",
        sem.oracle,
        launch.block.0,
        sem.oracle_reason,
        lineage,
        m.seed,
    );
    let _ = prediction;
    render_kernel_file(
        &doc,
        &m.kernel.extra_uses(),
        &m.kernel.extra_items(),
        &m.kernel.render_body(),
        launch,
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::ConstructionOracle;
    use crate::prediction::ExpectedStatic;

    const L: fn() -> Launch = || Launch::one_block(32);

    fn find_mutant(all: &[Mutant], id: &str) -> Mutant {
        all.iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("no mutant `{id}`; have {:?}", all.iter().map(|m| &m.id).collect::<Vec<_>>()))
            .clone()
    }

    #[test]
    fn every_seed_is_distinct_and_survives_enumeration() {
        let all = enumerate(0, L());
        assert_eq!(all.len(), seeds().len(), "no seed renders to another seed");
        assert!(all.iter().all(|m| m.depth() == 0));
    }

    #[test]
    fn dropping_the_guard_relabels_the_case_instead_of_inheriting_its_oracle() {
        // The brief's §11: a mutation that removes the bug must not leave the
        // case labelled as buggy.
        let all = mutations(&seed("barrier_divergent_intra_warp").unwrap(), L());
        let (_, k) = all
            .iter()
            .find(|(op, _)| op == "drop_guard@0")
            .expect("the guard can be dropped");
        let sem = interpret(k, L());
        assert_eq!(sem.oracle, ConstructionOracle::KnownSafe);
        assert_eq!(predict(&sem).expected, ExpectedStatic::Silent);
    }

    #[test]
    fn swapping_the_divergence_source_keeps_the_divergence_and_changes_the_tier() {
        let all = mutations(&seed("barrier_divergent_intra_warp").unwrap(), L());
        let (_, k) = all.iter().find(|(op, _)| op == "to_lanemask@0").unwrap();
        let sem = interpret(k, L());
        // lanemask_lt().count_ones() % 2 is still the lane's parity, so the
        // barrier is just as divergent...
        assert_eq!(sem.oracle, ConstructionOracle::KnownUnsafe);
        // ...but the analyzer documents that it cannot evaluate the source.
        assert_eq!(
            predict(&sem).expected,
            ExpectedStatic::WarningOnly { code: "RC001".into() }
        );
    }

    #[test]
    fn matching_the_mask_to_the_participants_produces_a_valid_program_that_is_not_gated() {
        let all = mutations(&seed("collective_under_divergence").unwrap(), L());
        let (_, k) = all
            .iter()
            .find(|(op, _)| op == "mask_match_participants@0.0")
            .expect("the participating lanes are known, so the mask can match them");
        let sem = interpret(k, L());
        assert_eq!(sem.oracle, ConstructionOracle::KnownMaskValid);
        assert!(k.render_body().contains("0x5555_5555"));
        // This case was built to try to force a false positive. It did not: see
        // model::tests and docs/stage-4.md.
        assert_eq!(
            predict(&sem).expected,
            ExpectedStatic::WarningOnly { code: "RC002".into() }
        );
    }

    #[test]
    fn a_truncating_cast_mutation_renders_the_cast() {
        let all = mutations(&seed("barrier_divergent_intra_warp").unwrap(), L());
        let (_, k) = all.iter().find(|(op, _)| op == "truncate_operand@0").unwrap();
        assert!(k.render_body().contains("(i.get() as u8) as u32 % 2 == 0"));
    }

    #[test]
    fn enumeration_is_deterministic_and_deduplicated() {
        let a = enumerate(1, L());
        let b = enumerate(1, L());
        assert_eq!(
            a.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
            b.iter().map(|m| m.id.clone()).collect::<Vec<_>>()
        );
        let prints: BTreeSet<String> = a.iter().map(|m| fingerprint(&m.kernel)).collect();
        assert_eq!(prints.len(), a.len(), "no two cases render the same program");
        assert!(a.len() > seeds().len() * 3, "depth 1 should be a real space, got {}", a.len());
    }

    #[test]
    fn sampling_is_reproducible_and_keeps_every_control() {
        let all = enumerate(1, L());
        let a = sample(all.clone(), 30, 7);
        let b = sample(all.clone(), 30, 7);
        assert_eq!(a.len(), 30);
        assert_eq!(
            a.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
            b.iter().map(|m| m.id.clone()).collect::<Vec<_>>()
        );
        assert_eq!(a.iter().filter(|m| m.depth() == 0).count(), seeds().len());
        let c = sample(all, 30, 8);
        assert_ne!(
            a.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
            c.iter().map(|m| m.id.clone()).collect::<Vec<_>>(),
            "a different seed must give a different subset"
        );
    }

    #[test]
    fn a_record_carries_the_computed_oracle_and_the_attributed_prediction() {
        let all = enumerate(1, L());
        let m = find_mutant(&all, "barrier_divergent_intra_warp+to_lanemask@0");
        let r = record(&m, 0, vec![L()]);
        assert_eq!(r.oracle, ConstructionOracle::KnownUnsafe);
        let basis = r.prediction_basis.unwrap();
        assert_eq!(basis.seed_template, "barrier_divergent_intra_warp");
        assert_eq!(basis.mutation_lineage, vec!["to_lanemask@0"]);
        assert!(r.kernel_source.contains("MUTATION: to_lanemask@0 of barrier_divergent_intra_warp"));
            assert!(
            !r.kernel_source.contains("RC001"),
            "the analyzed source must not carry the prediction: a rule change \
             would rename every case in the corpus"
        );
        assert_eq!(r.expected_static, crate::prediction::ExpectedStatic::WarningOnly { code: "RC001".into() });
        assert!(r.documented_limitation.is_some());
    }

    #[test]
    fn the_sibling_operators_differ_only_in_order_and_the_order_is_the_finding() {
        let all = mutations(&seed("barrier_divergent_intra_warp").unwrap(), L());
        let below = all.iter().find(|(op, _)| op == "add_lane_env_sibling@0.0").unwrap().1.clone();
        let above =
            all.iter().find(|(op, _)| op == "prepend_lane_env_sibling@0.0").unwrap().1.clone();
        assert_ne!(below, above, "the two operators must produce different programs");
        for k in [&below, &above] {
            assert!(k.render_body().contains("if i.get() % 2 == 0 {"));
            assert!(k.render_body().contains("if warp::warp_id() == 0 {"));
            assert_eq!(interpret(k, L()).oracle, ConstructionOracle::KnownUnsafe);
        }
        assert_eq!(
            predict(&interpret(&below, L())).expected,
            ExpectedStatic::Gating { code: "RC001".into() },
        );
        assert_eq!(
            predict(&interpret(&above, L())).expected,
            ExpectedStatic::WarningOnly { code: "RC001".into() },
            "the same barrier, moved above the confirmable one, un-gates it"
        );
    }

    #[test]
    fn the_lane_env_sibling_operator_leaves_the_original_guard_untouched() {
        // The point of this operator is that the confirmable barrier is not
        // modified at all: whatever changes must come from the addition.
        let all = mutations(&seed("barrier_divergent_intra_warp").unwrap(), L());
        let (_, k) = all
            .iter()
            .find(|(op, _)| op == "add_lane_env_sibling@0.0")
            .expect("a barrier inside a guard can get a sibling");
        let body = k.render_body();
        assert!(body.contains("if i.get() % 2 == 0 {"), "the original guard survives verbatim");
        assert!(body.contains("if warp::warp_id() == 0 {"));
        let sem = interpret(k, L());
        assert_eq!(sem.oracle, ConstructionOracle::KnownUnsafe);
        // This case was built to demonstrate that an added lane-environment
        // barrier un-gates its neighbour. It measured the opposite -- still
        // RC001/confirmed with a witness -- and that refutation is why the rule
        // is about program order and not about the function as a whole.
        assert_eq!(
            predict(&sem).expected,
            ExpectedStatic::Gating { code: "RC001".into() },
            "a lane-environment barrier *after* a confirmable one leaves it gated"
        );
    }

    #[test]
    fn cloning_a_guard_to_the_end_builds_the_a_b_a_shape() {
        // From `if A { sync }` plus a complementary sibling, cloning the first
        // guard to the end gives A, B, A -- three sites, two sources, and the
        // one shape that separates "shares the witnessed source" from "is an
        // unbroken prefix of it".
        let a_b = mutations(&seed("barrier_divergent_intra_warp").unwrap(), L())
            .into_iter()
            .find(|(op, _)| op == "complementary_guard@0")
            .unwrap()
            .1;
        let (_, a_b_a) = mutations(&a_b, L())
            .into_iter()
            .find(|(op, _)| op == "clone_guard_to_end@0.0")
            .expect("the first guard can be cloned to the end");
        let body = a_b_a.render_body();
        let guards: Vec<&str> = body
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("if ") && !l.contains("out.get_mut"))
            .collect();
        assert_eq!(
            guards,
            vec![
                "if i.get() % 2 == 0 {",
                "if !(i.get() % 2 == 0) {",
                "if i.get() % 2 == 0 {",
            ]
        );
        assert_eq!(interpret(&a_b_a, L()).oracle, ConstructionOracle::KnownUnsafe);
    }
}
