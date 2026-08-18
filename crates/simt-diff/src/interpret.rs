//! Exact execution of an [`crate::ir::Kernel`] over every thread of a launch.
//!
//! This is what lets the mutation engine keep the brief's §11 promise. A
//! mutated kernel never inherits a label: its oracle, its reference model and
//! its per-site facts are all recomputed here, from the mutant, by running it.
//!
//! Two modelling decisions are load-bearing, and both are conservative on
//! purpose:
//!
//! 1. **A barrier is divergent unless every thread of the block executes it
//!    the same number of times.** Complementary guards (`% 2 == 0` in one
//!    branch, `% 2 != 0` in the other) give every thread exactly one arrival,
//!    and arrival counting might well rescue them on real hardware. The CUDA
//!    programming model still calls a barrier under divergent control
//!    undefined, and a laboratory does not get to assume hardware charity.
//! 2. **A reference model is emitted only when execution is defined.** If any
//!    barrier is divergent, or any collective names a lane that is absent, the
//!    program has no defined result and claiming expected values would be
//!    inventing evidence -- the mistake `docs/research-baseline.md` §9.2
//!    records.

use std::collections::BTreeMap;

use crate::ir::{Kernel, Mask, Stmt, WriteExpr};
use crate::oracle::ConstructionOracle;
use crate::records::{Launch, ReferenceModel};

/// A hard cap on loop iterations. Bounds here are `% k` of a thread index, so
/// this is unreachable in practice; it exists so a mutation can never hang the
/// generator.
const MAX_ITERATIONS: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteKind {
    Barrier,
    Collective,
}

/// The enclosing control context of a site, in the terms the analyzer's
/// documentation uses. Purely static: this is what a static analyzer could
/// see, and keeping it separate from the executed facts is what makes the
/// comparison in [`crate::model`] meaningful rather than circular.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuardChain {
    /// Some enclosing guard or loop bound reads a per-thread value, so the
    /// site is not provably reached uniformly.
    pub statically_divergent: bool,
    /// Some enclosing condition reads the lane environment (`warp_id`,
    /// `lanemask_*`), which the analyzer documents it cannot evaluate.
    pub lane_env: bool,
    /// Some enclosing condition is thread-index arithmetic with literal
    /// operands, which the 32-lane witness interpreter can replay.
    pub index_evaluable: bool,
    /// Some enclosing condition needs a truncating cast to evaluate.
    pub trunc_cast: bool,
    /// The site is inside a loop whose trip count is per-thread.
    pub divergent_loop: bool,
    pub nesting: usize,
    /// 0 when the site is lexically in the kernel; otherwise how many calls
    /// away it is.
    pub via_helper_depth: usize,
    /// The site is somewhere inside a loop.
    pub enclosing_loop: bool,
    /// A per-thread `if` encloses the site *and* is itself inside a loop. This
    /// is not a shape any documentation mentions; it earns a field because
    /// measurement showed it decides whether a finding is witness-promoted.
    pub divergent_guard_inside_loop: bool,
    /// The enclosing per-thread conditions, rendered and joined. Two sites with
    /// the same string are divergent for the same reason, which measurement
    /// showed decides whether the second one is promoted along with the first.
    pub divergence_source: Option<String>,
}

/// One barrier or collective program point, with everything known about it.
#[derive(Clone, Debug)]
pub struct Site {
    pub key: String,
    pub kind: SiteKind,
    pub guards: GuardChain,
    pub mask: Option<Mask>,
    /// Executions per thread id. Absent threads executed it zero times.
    pub counts: BTreeMap<u32, u32>,
    /// Barriers: threads disagree on how many times they reach it.
    pub divergent: bool,
    /// Collectives, per warp: the lanes present, and the lanes the mask names.
    pub participants: BTreeMap<u32, u32>,
    pub named: BTreeMap<u32, u32>,
    /// Collectives: the mask names exactly the participating lanes.
    pub mask_valid: Option<bool>,
    /// Collectives: every named lane is present, so the operation has a
    /// defined reading even if the mask is not exactly the participants.
    pub value_defined: bool,
}

impl Site {
    pub fn executed(&self) -> bool {
        self.counts.values().any(|c| *c > 0)
    }

    pub fn threads_executing(&self) -> Vec<u32> {
        self.counts
            .iter()
            .filter(|(_, c)| **c > 0)
            .map(|(t, _)| *t)
            .collect()
    }
}

/// Everything the generator knows about a kernel at a launch, computed.
#[derive(Clone, Debug)]
pub struct Semantics {
    pub launch: Launch,
    pub sites: Vec<Site>,
    pub oracle: ConstructionOracle,
    pub oracle_reason: String,
    pub reference: Option<ReferenceModel>,
    /// Why execution is undefined, when it is. Empty means defined.
    pub undefined: Vec<String>,
}

impl Semantics {
    pub fn barriers(&self) -> impl Iterator<Item = &Site> {
        self.sites
            .iter()
            .filter(|s| s.kind == SiteKind::Barrier && s.executed())
    }

    pub fn collectives(&self) -> impl Iterator<Item = &Site> {
        self.sites
            .iter()
            .filter(|s| s.kind == SiteKind::Collective && s.executed())
    }

    pub fn has_divergent_barrier(&self) -> bool {
        self.barriers().any(|s| s.divergent)
    }

    pub fn has_invalid_mask(&self) -> bool {
        self.collectives().any(|s| s.mask_valid == Some(false))
    }
}

pub fn interpret(kernel: &Kernel, launch: Launch) -> Semantics {
    let threads = launch.block.0.max(1) * launch.block.1.max(1) * launch.block.2.max(1);

    // ---- static pass: the sites and their control context ---------------
    let mut plan: Vec<PlannedSite> = Vec::new();
    let mut loop_index: BTreeMap<String, usize> = BTreeMap::new();
    collect(
        &kernel.stmts,
        &mut Vec::new(),
        GuardChain::default(),
        kernel,
        &mut plan,
        &mut loop_index,
    );

    // ---- dynamic pass: run every thread ----------------------------------
    let mut counts: BTreeMap<String, BTreeMap<u32, u32>> = BTreeMap::new();
    let mut participants: BTreeMap<String, BTreeMap<u32, u32>> = BTreeMap::new();
    let mut last_ballot: BTreeMap<u32, String> = BTreeMap::new();
    let mut loop_final: BTreeMap<u32, u32> = BTreeMap::new();
    let loop_count = loop_index.len();

    for tid in 0..threads {
        let mut run = ThreadRun {
            tid,
            counters: vec![0; loop_count],
            counts: BTreeMap::new(),
            last_ballot: None,
        };
        exec(&kernel.stmts, &mut Vec::new(), &loop_index, &mut run);
        for (key, c) in run.counts {
            let entry = counts.entry(key.clone()).or_default();
            *entry.entry(tid).or_insert(0) += c;
            if key.starts_with("collective") {
                let e = participants.entry(key).or_default();
                *e.entry(tid / 32).or_insert(0) |= 1u32 << (tid % 32);
            }
        }
        if let Some(k) = run.last_ballot {
            last_ballot.insert(tid, k);
        }
        loop_final.insert(tid, run.counters.first().copied().unwrap_or(0));
    }

    // ---- assemble the sites ---------------------------------------------
    let mut sites = Vec::new();
    for p in &plan {
        let mut counts_full: BTreeMap<u32, u32> = (0..threads).map(|t| (t, 0)).collect();
        if let Some(c) = counts.get(&p.key) {
            for (t, n) in c {
                counts_full.insert(*t, *n);
            }
        }
        let uniform = counts_full
            .values()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            <= 1;
        let executed = counts_full.values().any(|c| *c > 0);

        let (mut named, mut valid, mut value_defined) = (BTreeMap::new(), None, true);
        let parts = participants.get(&p.key).cloned().unwrap_or_default();
        if p.kind == SiteKind::Collective && executed {
            let mut all_valid = true;
            let mut all_defined = true;
            for (warp, present) in &parts {
                let n = match p.mask.and_then(Mask::named_lanes) {
                    Some(v) => v,
                    // active_mask() names the participants by definition.
                    None => *present,
                };
                named.insert(*warp, n);
                if n != *present {
                    all_valid = false;
                }
                if n & !*present != 0 {
                    all_defined = false;
                }
            }
            valid = Some(all_valid);
            value_defined = all_defined;
        }

        sites.push(Site {
            key: p.key.clone(),
            kind: p.kind,
            guards: p.guards.clone(),
            mask: p.mask,
            counts: counts_full,
            divergent: p.kind == SiteKind::Barrier && executed && !uniform,
            participants: parts,
            named,
            mask_valid: valid,
            value_defined,
        });
    }

    // ---- undefined-ness, then the oracle --------------------------------
    let mut undefined = Vec::new();
    for s in sites.iter().filter(|s| s.executed()) {
        if s.kind == SiteKind::Barrier && s.divergent {
            let reaching = s.threads_executing().len();
            undefined.push(format!(
                "the barrier at {} is executed by {reaching} of {threads} threads, \
                 and threads that disagree at a block-wide barrier make the \
                 launch undefined",
                s.key
            ));
        }
        if s.kind == SiteKind::Collective && !s.value_defined {
            undefined.push(format!(
                "the collective at {} names lanes that are not present, so the \
                 operation has no defined result",
                s.key
            ));
        }
    }

    let (oracle, oracle_reason) = derive_oracle(&sites, threads);
    let reference = if undefined.is_empty() {
        reference_model(kernel, launch, threads, &sites, &last_ballot, &loop_final)
    } else {
        None
    };

    Semantics {
        launch,
        sites,
        oracle,
        oracle_reason,
        reference,
        undefined,
    }
}

fn derive_oracle(sites: &[Site], threads: u32) -> (ConstructionOracle, String) {
    if let Some(s) = sites
        .iter()
        .find(|s| s.executed() && s.kind == SiteKind::Barrier && s.divergent)
    {
        let reaching = s.threads_executing().len();
        return (
            ConstructionOracle::KnownUnsafe,
            format!(
                "the barrier at {} is reached by {reaching} of {threads} threads \
                 of the block, so threads disagree at a block-wide barrier",
                s.key
            ),
        );
    }
    if let Some(s) = sites
        .iter()
        .find(|s| s.executed() && s.kind == SiteKind::Collective && s.mask_valid == Some(false))
    {
        let (named, present) = describe_mask(s);
        return (
            ConstructionOracle::KnownMaskInvalid,
            format!(
                "the collective at {} runs with {present} lane(s) present while \
                 its mask names {named}, so the named set and the participating \
                 set are not the same",
                s.key
            ),
        );
    }
    if let Some(s) = sites
        .iter()
        .find(|s| s.executed() && s.kind == SiteKind::Collective)
    {
        let (named, present) = describe_mask(s);
        return (
            ConstructionOracle::KnownMaskValid,
            format!(
                "the collective at {} is reached by {present} lane(s) and its \
                 mask names {named}, which are the same lanes",
                s.key
            ),
        );
    }
    if sites.iter().any(|s| s.executed()) {
        return (
            ConstructionOracle::KnownSafe,
            format!(
                "every one of the {threads} threads of the block reaches every \
                 barrier the same number of times"
            ),
        );
    }
    (
        ConstructionOracle::KnownSafe,
        format!(
            "the kernel contains no barrier or collective reachable by any of the {threads} threads"
        ),
    )
}

fn describe_mask(s: &Site) -> (String, String) {
    let named = s
        .named
        .values()
        .map(|v| format!("{} lane(s)", v.count_ones()))
        .collect::<Vec<_>>()
        .join("/");
    let present = s
        .participants
        .values()
        .map(|v| v.count_ones().to_string())
        .collect::<Vec<_>>()
        .join("/");
    (
        if named.is_empty() {
            "no lanes".to_string()
        } else {
            named
        },
        if present.is_empty() {
            "0".to_string()
        } else {
            present
        },
    )
}

fn reference_model(
    kernel: &Kernel,
    launch: Launch,
    threads: u32,
    sites: &[Site],
    last_ballot: &BTreeMap<u32, String>,
    loop_final: &BTreeMap<u32, u32>,
) -> Option<ReferenceModel> {
    let by_key: BTreeMap<&str, &Site> = sites.iter().map(|s| (s.key.as_str(), s)).collect();
    let mut expected = BTreeMap::new();
    let description = match kernel.write_expr() {
        WriteExpr::One => {
            for t in 0..threads {
                expected.insert(t, 1u32);
            }
            "every thread reaches the write, so every lane writes 1".to_string()
        }
        WriteExpr::LoopCounter => {
            for t in 0..threads {
                expected.insert(t, *loop_final.get(&t).unwrap_or(&0));
            }
            "each lane writes the number of iterations its own trip count gave it".to_string()
        }
        WriteExpr::Ballot => {
            for t in 0..threads {
                let value = match last_ballot.get(&t) {
                    Some(key) => {
                        let site = by_key.get(key.as_str())?;
                        // Executed more than once by some thread: participation
                        // is then per-iteration, and this interpreter only
                        // records the union. Rather than predict a value from a
                        // set that was never simultaneously present, predict
                        // nothing -- the §9.2 lesson.
                        if site.counts.values().any(|c| *c > 1) {
                            return None;
                        }
                        *site.named.get(&(t / 32))?
                    }
                    // `let mut b = 0u32` is what the lane keeps if it never
                    // reaches the collective.
                    None => 0,
                };
                expected.insert(t, value);
            }
            "each lane writes the ballot over the lanes its mask names, which is \
             defined here because every named lane is present"
                .to_string()
        }
    };
    Some(ReferenceModel {
        description,
        expected,
        launch,
    })
}

// ------------------------------------------------------------ static pass ---

struct PlannedSite {
    key: String,
    kind: SiteKind,
    guards: GuardChain,
    mask: Option<Mask>,
}

fn key_of(prefix: &str, path: &[usize]) -> String {
    format!(
        "{prefix}@{}",
        path.iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(".")
    )
}

fn collect(
    stmts: &[Stmt],
    path: &mut Vec<usize>,
    chain: GuardChain,
    kernel: &Kernel,
    out: &mut Vec<PlannedSite>,
    loop_index: &mut BTreeMap<String, usize>,
) {
    for (i, s) in stmts.iter().enumerate() {
        path.push(i);
        match s {
            Stmt::Barrier => out.push(PlannedSite {
                key: key_of("barrier", path),
                kind: SiteKind::Barrier,
                guards: chain.clone(),
                mask: None,
            }),
            Stmt::Ballot { mask } => out.push(PlannedSite {
                key: key_of("collective", path),
                kind: SiteKind::Collective,
                guards: chain.clone(),
                mask: Some(*mask),
            }),
            Stmt::CallHelper => {
                // The call site is where this thread's arrival happens, so it
                // is the site. Keying the helper's barrier once globally would
                // make two complementary call sites look uniform while two
                // inlined barriers look divergent -- the same program, two
                // labels.
                let mut guards = chain.clone();
                guards.via_helper_depth = kernel.helper_depth.max(1);
                out.push(PlannedSite {
                    key: key_of("call", path),
                    kind: SiteKind::Barrier,
                    guards,
                    mask: None,
                });
            }
            Stmt::If { pred, body } => {
                let mut inner = chain.clone();
                inner.nesting += 1;
                let per_thread = pred.reads_lane_index() || pred.reads_lane_environment();
                inner.divergent_guard_inside_loop |= per_thread && chain.enclosing_loop;
                inner.statically_divergent |= per_thread;
                if per_thread {
                    inner.divergence_source = Some(match &chain.divergence_source {
                        Some(outer) => format!("{outer} && {}", pred.render()),
                        None => pred.render(),
                    });
                }
                inner.lane_env |= pred.reads_lane_environment();
                inner.trunc_cast |= pred.has_truncating_cast();
                inner.index_evaluable |= pred.reads_lane_index() && !pred.has_truncating_cast();
                collect(body, path, inner, kernel, out, loop_index);
            }
            Stmt::Loop { bound, body } => {
                loop_index.insert(key_of("loop", path), loop_index.len());
                let mut inner = chain.clone();
                inner.nesting += 1;
                inner.enclosing_loop = true;
                let per_thread = bound.reads_lane_index() || bound.reads_lane_environment();
                inner.statically_divergent |= per_thread;
                inner.divergent_loop |= per_thread;
                if per_thread {
                    let trip = format!("trip count {}", bound.render(crate::ir::Ctx::U32));
                    inner.divergence_source = Some(match &chain.divergence_source {
                        Some(outer) => format!("{outer} && {trip}"),
                        None => trip,
                    });
                }
                inner.lane_env |= bound.reads_lane_environment();
                inner.trunc_cast |= bound.has_truncating_cast();
                inner.index_evaluable |= bound.reads_lane_index() && !bound.has_truncating_cast();
                collect(body, path, inner, kernel, out, loop_index);
            }
        }
        path.pop();
    }
}

// ----------------------------------------------------------- dynamic pass ---

struct ThreadRun {
    tid: u32,
    counters: Vec<u32>,
    counts: BTreeMap<String, u32>,
    last_ballot: Option<String>,
}

fn exec(
    stmts: &[Stmt],
    path: &mut Vec<usize>,
    loop_index: &BTreeMap<String, usize>,
    run: &mut ThreadRun,
) {
    for (i, s) in stmts.iter().enumerate() {
        path.push(i);
        match s {
            Stmt::Barrier => {
                *run.counts.entry(key_of("barrier", path)).or_insert(0) += 1;
            }
            Stmt::CallHelper => {
                *run.counts.entry(key_of("call", path)).or_insert(0) += 1;
            }
            Stmt::Ballot { .. } => {
                let key = key_of("collective", path);
                *run.counts.entry(key.clone()).or_insert(0) += 1;
                run.last_ballot = Some(key);
            }
            Stmt::If { pred, body } => {
                if pred.eval(run.tid) {
                    exec(body, path, loop_index, run);
                }
            }
            Stmt::Loop { bound, body } => {
                let idx = *loop_index.get(&key_of("loop", path)).unwrap_or(&0);
                let limit = bound.eval(run.tid).min(MAX_ITERATIONS);
                if let Some(c) = run.counters.get_mut(idx) {
                    *c = 0;
                }
                while run.counters.get(idx).copied().unwrap_or(0) < limit {
                    exec(body, path, loop_index, run);
                    if let Some(c) = run.counters.get_mut(idx) {
                        *c += 1;
                    } else {
                        break;
                    }
                }
            }
        }
        path.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{CmpOp, Pred, Value};

    fn even() -> Pred {
        Pred::Cmp(Value::Rem(Box::new(Value::LaneIndex), 2), CmpOp::Eq, 0)
    }

    fn odd() -> Pred {
        Pred::Cmp(Value::Rem(Box::new(Value::LaneIndex), 2), CmpOp::Ne, 0)
    }

    #[test]
    fn an_unguarded_barrier_is_safe_and_has_a_reference_model() {
        let s = interpret(&Kernel::new(vec![Stmt::Barrier]), Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownSafe);
        assert!(s.undefined.is_empty());
        assert_eq!(s.reference.unwrap().expected[&0], 1);
    }

    #[test]
    fn a_guarded_barrier_is_unsafe_and_ships_no_reference_model() {
        let k = Kernel::new(vec![Stmt::If {
            pred: even(),
            body: vec![Stmt::Barrier],
        }]);
        let s = interpret(&k, Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownUnsafe);
        assert!(
            s.reference.is_none(),
            "undefined execution has no expected values"
        );
        assert!(s.oracle_reason.contains("16 of 32"));
    }

    #[test]
    fn complementary_guards_are_still_reported_unsafe() {
        // Every thread arrives exactly once, and arrival counting might rescue
        // this on hardware. The programming model does not, so neither do we.
        let k = Kernel::new(vec![
            Stmt::If {
                pred: even(),
                body: vec![Stmt::Barrier],
            },
            Stmt::If {
                pred: odd(),
                body: vec![Stmt::Barrier],
            },
        ]);
        let s = interpret(&k, Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownUnsafe);
    }

    #[test]
    fn a_guard_that_is_true_for_every_thread_leaves_the_barrier_uniform() {
        // `i.get() % 1 == 0` holds everywhere: statically a per-thread guard,
        // dynamically no divergence at all. Exactly the case a 32-lane witness
        // replay would fail to confirm.
        let k = Kernel::new(vec![Stmt::If {
            pred: Pred::Cmp(Value::Rem(Box::new(Value::LaneIndex), 1), CmpOp::Eq, 0),
            body: vec![Stmt::Barrier],
        }]);
        let s = interpret(&k, Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownSafe);
        assert!(s.sites.iter().any(|x| x.guards.statically_divergent));
    }

    #[test]
    fn warp_id_divergence_depends_on_the_launch() {
        let k = Kernel::new(vec![Stmt::If {
            pred: Pred::Cmp(Value::WarpId, CmpOp::Eq, 0),
            body: vec![Stmt::Barrier],
        }]);
        // One warp: the guard is true for every thread, so nothing diverges.
        assert_eq!(
            interpret(&k, Launch::one_block(32)).oracle,
            ConstructionOracle::KnownSafe
        );
        // Two warps: a whole warp skips a block-wide barrier.
        assert_eq!(
            interpret(&k, Launch::one_block(64)).oracle,
            ConstructionOracle::KnownUnsafe
        );
    }

    #[test]
    fn a_full_mask_at_a_convergent_call_is_valid_and_predicts_the_value() {
        let k = Kernel::new(vec![Stmt::Ballot {
            mask: Mask::Literal(0xffff_ffff),
        }]);
        let s = interpret(&k, Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownMaskValid);
        assert_eq!(s.reference.unwrap().expected[&7], 0xffff_ffff);
    }

    #[test]
    fn a_shrunk_mask_is_invalid_but_still_has_a_defined_reading() {
        let k = Kernel::new(vec![Stmt::Ballot {
            mask: Mask::Literal(0x0000_ffff),
        }]);
        let s = interpret(&k, Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownMaskInvalid);
        // Every named lane is present, so the mask it was handed still defines
        // a value -- which is the only thing that made the Stage 1 bug visible.
        let r = s.reference.expect("a defined reading exists");
        assert_eq!(r.expected[&0], 0x0000_ffff);
    }

    #[test]
    fn a_full_mask_under_divergence_has_no_defined_reading() {
        let k = Kernel::new(vec![Stmt::If {
            pred: even(),
            body: vec![Stmt::Ballot {
                mask: Mask::Literal(0xffff_ffff),
            }],
        }]);
        let s = interpret(&k, Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownMaskInvalid);
        assert!(s.reference.is_none());
        assert!(s.undefined[0].contains("names lanes that are not present"));
    }

    #[test]
    fn a_mask_matching_the_participants_is_valid_under_divergence() {
        // The correct guarded partial-warp idiom, spelled with a literal.
        let k = Kernel::new(vec![Stmt::If {
            pred: even(),
            body: vec![Stmt::Ballot {
                mask: Mask::Literal(0x5555_5555),
            }],
        }]);
        let s = interpret(&k, Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownMaskValid);
        let r = s.reference.expect("valid, so the value is defined");
        assert_eq!(r.expected[&0], 0x5555_5555, "an even lane reads the ballot");
        assert_eq!(
            r.expected[&1], 0,
            "an odd lane never reaches it and keeps 0"
        );
    }

    #[test]
    fn active_mask_is_valid_by_definition_wherever_it_is_called() {
        let k = Kernel::new(vec![Stmt::If {
            pred: even(),
            body: vec![Stmt::Ballot {
                mask: Mask::ActiveMask,
            }],
        }]);
        let s = interpret(&k, Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownMaskValid);
        assert_eq!(s.reference.unwrap().expected[&0], 0x5555_5555);
    }

    #[test]
    fn a_divergent_trip_count_diverges_the_barrier_inside_the_loop() {
        let k = Kernel::new(vec![Stmt::Loop {
            bound: Value::Rem(Box::new(Value::LaneIndex), 4),
            body: vec![Stmt::Barrier],
        }]);
        let s = interpret(&k, Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownUnsafe);
        let site = s
            .sites
            .iter()
            .find(|x| x.kind == SiteKind::Barrier)
            .unwrap();
        assert_eq!(site.counts[&0], 0);
        assert_eq!(site.counts[&3], 3);
        assert!(site.guards.divergent_loop);
    }

    #[test]
    fn a_barrier_behind_a_helper_carries_the_call_sites_context() {
        let k = Kernel::with_helper(
            vec![Stmt::If {
                pred: even(),
                body: vec![Stmt::CallHelper],
            }],
            1,
        );
        let s = interpret(&k, Launch::one_block(32));
        let site = s.sites.iter().find(|x| x.key == "call@0.0").unwrap();
        assert_eq!(site.guards.via_helper_depth, 1);
        assert!(site.guards.index_evaluable);
        assert!(site.divergent);
        assert_eq!(s.oracle, ConstructionOracle::KnownUnsafe);
    }

    #[test]
    fn an_unguarded_helper_call_leaves_the_barrier_uniform() {
        let k = Kernel::with_helper(vec![Stmt::CallHelper], 1);
        let s = interpret(&k, Launch::one_block(32));
        assert_eq!(s.oracle, ConstructionOracle::KnownSafe);
        assert!(s.reference.is_some());
    }
}
