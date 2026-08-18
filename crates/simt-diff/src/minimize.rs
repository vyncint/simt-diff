//! Shrink a case while keeping what makes it interesting (brief §21).
//!
//! Delta debugging is usually delicate work on a text file, where every
//! candidate might not even parse and a "smaller" version might not mean the
//! same thing. Neither problem exists here: candidates are [`crate::ir::Kernel`]
//! values, so they always render to a valid program, and their oracle is
//! recomputed by [`crate::interpret`] rather than assumed to survive.
//!
//! That is what makes the property being preserved worth stating precisely. The
//! findings in `docs/stage-4.md` are not crashes; they are *observations* --
//! "RC001 at warning tier with no witness, on a kernel construction says is
//! unsafe". A reduction that quietly turned the kernel safe would preserve the
//! observation and destroy the finding, so the default property pins the
//! construction oracle as well as the analyzer's answer.

use std::path::{Path, PathBuf};

use crate::analyzer::Analyzer;
use crate::interpret::{Semantics, interpret};
use crate::ir::{Kernel, Mask, Pred, Stmt, Value};
use crate::oracle::ConstructionOracle;
use crate::records::{AnalyzerRecord, Launch};

/// What a reduction must not change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Property {
    /// The construction oracle and the analyzer's signature both stay as they
    /// were. This is the default, and the only one that is safe without thinking
    /// about the case first.
    OracleAndSignature {
        oracle: ConstructionOracle,
        signature: String,
    },
    /// Only the analyzer's signature is pinned. Useful when the point of the
    /// case is what the analyzer says and not what the kernel means -- but it
    /// will happily reduce a bug into a different kernel that reads the same.
    Signature { signature: String },
}

impl Property {
    /// Read the property off the case being minimized, which is what "keep this
    /// interesting" means when nobody said otherwise.
    pub fn of(sem: &Semantics, analysis: &AnalyzerRecord) -> Property {
        Property::OracleAndSignature {
            oracle: sem.oracle,
            signature: analysis.signature(),
        }
    }

    pub fn holds(&self, sem: &Semantics, analysis: &AnalyzerRecord) -> bool {
        match self {
            Property::OracleAndSignature { oracle, signature } => {
                sem.oracle == *oracle && &analysis.signature() == signature
            }
            Property::Signature { signature } => &analysis.signature() == signature,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Property::OracleAndSignature { oracle, signature } => {
                format!("oracle {oracle:?} and analyzer signature {signature}")
            }
            Property::Signature { signature } => format!("analyzer signature {signature}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub kernel: Kernel,
    /// Every accepted reduction, in order.
    pub accepted: Vec<String>,
    /// Every rejected one, with why. A minimizer that only reports what worked
    /// hides the shape of the boundary it found.
    pub rejected: Vec<String>,
    pub start_size: usize,
    pub end_size: usize,
    pub analyses: usize,
}

pub struct Minimizer<'a> {
    pub analyzer: &'a dyn Analyzer,
    /// Scratch directory for the candidate crates.
    pub workdir: PathBuf,
    pub launch: Launch,
    pub property: Property,
    /// Emitted so a candidate can be analyzed at all.
    pub device_dep: crate::emit::DeviceDep,
}

impl Minimizer<'_> {
    /// Greedy fixpoint: keep applying the first reduction that preserves the
    /// property, until none does.
    ///
    /// Greedy rather than a full ddmin pass because every candidate costs an
    /// analyzer run of a few seconds, and these kernels are small enough that
    /// the difference in result is nil.
    pub fn run(&self, start: &Kernel) -> Result<Outcome, String> {
        let start_size = size(start);
        let mut current = start.clone();
        let mut accepted = Vec::new();
        let mut rejected = Vec::new();
        let mut analyses = 0usize;

        loop {
            let mut progressed = false;
            for (name, candidate) in reductions(&current) {
                let sem = interpret(&candidate, self.launch);
                let analysis = self.analyze(&candidate, analyses)?;
                analyses += 1;
                if self.property.holds(&sem, &analysis) {
                    accepted.push(name);
                    current = candidate;
                    progressed = true;
                    break;
                }
                rejected.push(format!(
                    "{name}: oracle {:?}, signature {}",
                    sem.oracle,
                    analysis.signature()
                ));
            }
            if !progressed {
                break;
            }
        }

        Ok(Outcome {
            kernel: current.clone(),
            accepted,
            rejected,
            start_size,
            end_size: size(&current),
            analyses,
        })
    }

    fn analyze(&self, kernel: &Kernel, n: usize) -> Result<AnalyzerRecord, String> {
        let dir = self.workdir.join(format!("candidate-{n:03}"));
        let record = crate::mutate::record_for_kernel(kernel, "candidate", self.launch);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        crate::emit::emit(
            &dir,
            &record,
            &crate::emit::DeviceDep::default(),
            &self.device_dep,
        )
        .map_err(|e| e.to_string())?;
        self.analyzer
            .analyze(&dir.join("kernel"))
            .map_err(|e| format!("analyzing {}: {e}", dir.display()))
    }
}

/// Node count: statements, plus the predicate and value nodes inside them, plus
/// one per helper level. The reductions below all decrease it, which is what
/// makes the fixpoint terminate.
pub fn size(kernel: &Kernel) -> usize {
    fn stmt_size(s: &Stmt) -> usize {
        match s {
            Stmt::Barrier | Stmt::CallHelper => 1,
            Stmt::Ballot { mask } => 1 + usize::from(!matches!(mask, Mask::Literal(_))),
            Stmt::If { pred, body } => {
                1 + pred_size(pred) + body.iter().map(stmt_size).sum::<usize>()
            }
            Stmt::Loop { bound, body } => {
                1 + value_size(bound) + body.iter().map(stmt_size).sum::<usize>()
            }
        }
    }
    kernel.helper_depth + kernel.stmts.iter().map(stmt_size).sum::<usize>()
}

fn pred_size(p: &Pred) -> usize {
    match p {
        Pred::Cmp(v, _, _) => 1 + value_size(v),
        Pred::Not(inner) => 1 + pred_size(inner),
        Pred::And(a, b) => 1 + pred_size(a) + pred_size(b),
    }
}

fn value_size(v: &Value) -> usize {
    match v {
        Value::LaneIndex | Value::WarpId | Value::LaneMaskLtPopcount | Value::Const(_) => 1,
        Value::Rem(inner, _)
        | Value::Div(inner, _)
        | Value::BitAnd(inner, _)
        | Value::TruncU8(inner) => 1 + value_size(inner),
    }
}

/// Every single-step simplification of `kernel`, each strictly smaller.
///
/// Ordered largest-effect-first: deleting a statement says more about what the
/// finding needs than dropping a `!`.
pub fn reductions(kernel: &Kernel) -> Vec<(String, Kernel)> {
    let mut out: Vec<(String, Kernel)> = Vec::new();
    let sites: Vec<(Vec<usize>, Stmt)> = kernel
        .walk()
        .into_iter()
        .map(|(p, s)| (p, s.clone()))
        .collect();

    let mut push = |name: String, k: Kernel| {
        if size(&k) < size(kernel) {
            out.push((name, k));
        }
    };

    // Whole statements first.
    for (path, _) in &sites {
        let tag = path
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(".");
        let mut k = kernel.clone();
        if let Some((vec, idx)) = k.locate_mut(path) {
            vec.remove(idx);
        }
        push(format!("delete@{tag}"), k);
    }

    // Then the wrappers, keeping what they contained.
    for (path, stmt) in &sites {
        let tag = path
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(".");
        let body = match stmt {
            Stmt::If { body, .. } | Stmt::Loop { body, .. } => body.clone(),
            _ => continue,
        };
        let mut k = kernel.clone();
        if let Some((vec, idx)) = k.locate_mut(path) {
            vec.remove(idx);
            for (j, s) in body.into_iter().enumerate() {
                vec.insert(idx + j, s);
            }
        }
        let what = if matches!(stmt, Stmt::If { .. }) {
            "unwrap_guard"
        } else {
            "unwrap_loop"
        };
        push(format!("{what}@{tag}"), k);
    }

    // Then the conditions inside them.
    for (path, stmt) in &sites {
        let tag = path
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(".");
        if let Stmt::If { pred, .. } = stmt {
            for (name, simpler) in simplify_pred(pred) {
                let mut k = kernel.clone();
                if let Some((vec, idx)) = k.locate_mut(path)
                    && let Stmt::If { pred, .. } = &mut vec[idx]
                {
                    *pred = simpler;
                }
                push(format!("{name}@{tag}"), k);
            }
        }
        if let Stmt::Loop { bound, .. } = stmt {
            for (name, simpler) in simplify_value(bound) {
                let mut k = kernel.clone();
                if let Some((vec, idx)) = k.locate_mut(path)
                    && let Stmt::Loop { bound, .. } = &mut vec[idx]
                {
                    *bound = simpler;
                }
                push(format!("{name}_bound@{tag}"), k);
            }
        }
        if let Stmt::Ballot { mask } = stmt
            && !matches!(mask, Mask::Literal(_))
        {
            let mut k = kernel.clone();
            if let Some((vec, idx)) = k.locate_mut(path)
                && let Stmt::Ballot { mask } = &mut vec[idx]
            {
                *mask = Mask::Literal(0xffff_ffff);
            }
            push(format!("mask_to_literal@{tag}"), k);
        }
    }

    if kernel.helper_depth > 0 {
        let mut k = kernel.clone();
        k.helper_depth -= 1;
        // Dropping the last level would leave `barrier_helper` undefined, so the
        // call sites become the barrier itself.
        if k.helper_depth == 0 {
            inline_calls(&mut k.stmts);
        }
        push("shallower_helper".to_string(), k);
    }

    out
}

fn inline_calls(stmts: &mut [Stmt]) {
    for s in stmts.iter_mut() {
        match s {
            Stmt::CallHelper => *s = Stmt::Barrier,
            Stmt::If { body, .. } | Stmt::Loop { body, .. } => inline_calls(body),
            _ => {}
        }
    }
}

fn simplify_pred(p: &Pred) -> Vec<(String, Pred)> {
    match p {
        Pred::Not(inner) => vec![("drop_not".to_string(), (**inner).clone())],
        Pred::And(a, b) => vec![
            ("keep_left".to_string(), (**a).clone()),
            ("keep_right".to_string(), (**b).clone()),
        ],
        Pred::Cmp(v, op, rhs) => simplify_value(v)
            .into_iter()
            .map(|(name, simpler)| (name, Pred::Cmp(simpler, *op, *rhs)))
            .collect(),
    }
}

fn simplify_value(v: &Value) -> Vec<(String, Value)> {
    match v {
        Value::TruncU8(inner) => vec![("drop_cast".to_string(), (**inner).clone())],
        Value::Rem(inner, _) => vec![("drop_rem".to_string(), (**inner).clone())],
        Value::Div(inner, _) => vec![("drop_div".to_string(), (**inner).clone())],
        Value::BitAnd(inner, _) => vec![("drop_and".to_string(), (**inner).clone())],
        Value::LaneIndex | Value::WarpId | Value::LaneMaskLtPopcount | Value::Const(_) => vec![],
    }
}

/// Where a minimized case is written, for a human to read.
pub fn write_report(dir: &Path, outcome: &Outcome, property: &Property) -> std::io::Result<()> {
    let mut text = String::new();
    text.push_str("# Minimized case\n\n");
    text.push_str(&format!("Property preserved: {}\n\n", property.describe()));
    text.push_str(&format!(
        "Size {} -> {} nodes, in {} accepted reduction(s) over {} analyzer run(s).\n\n",
        outcome.start_size,
        outcome.end_size,
        outcome.accepted.len(),
        outcome.analyses
    ));
    text.push_str("## Accepted\n\n");
    for a in &outcome.accepted {
        text.push_str(&format!("- {a}\n"));
    }
    if outcome.accepted.is_empty() {
        text.push_str("- none: the case was already minimal under these reductions\n");
    }
    text.push_str(
        "\n## Rejected\n\nWhat each rejected reduction turned the case into. This is the \
         shape of the boundary: every line is a program that is *almost* this \
         finding and is not.\n\n",
    );
    for r in &outcome.rejected {
        text.push_str(&format!("- {r}\n"));
    }
    text.push_str("\n## The kernel\n\n```rust\n");
    text.push_str(&outcome.kernel.render_body());
    text.push_str("\n```\n");
    std::fs::write(dir.join("MINIMIZED.md"), text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::CmpOp;

    fn even() -> Pred {
        Pred::Cmp(Value::Rem(Box::new(Value::LaneIndex), 2), CmpOp::Eq, 0)
    }

    #[test]
    fn every_reduction_is_strictly_smaller() {
        // The termination argument, as a test: a greedy fixpoint over reductions
        // that do not shrink would not terminate.
        let k = Kernel::with_helper(
            vec![
                Stmt::If {
                    pred: Pred::And(Box::new(even()), Box::new(even())),
                    body: vec![Stmt::Loop {
                        bound: Value::Rem(Box::new(Value::TruncU8(Box::new(Value::LaneIndex))), 4),
                        body: vec![
                            Stmt::CallHelper,
                            Stmt::Ballot {
                                mask: Mask::ActiveMask,
                            },
                        ],
                    }],
                },
                Stmt::Barrier,
            ],
            2,
        );
        let before = size(&k);
        let rs = reductions(&k);
        assert!(rs.len() > 8, "only {} reductions offered", rs.len());
        for (name, candidate) in rs {
            assert!(
                size(&candidate) < before,
                "{name} did not shrink the case ({} -> {})",
                before,
                size(&candidate)
            );
        }
    }

    #[test]
    fn a_minimal_case_offers_no_reduction_that_keeps_it_meaningful() {
        // One guarded barrier: deleting either part is possible, but there is
        // nothing left to unwrap or simplify beyond the modulus.
        let k = Kernel::new(vec![Stmt::If {
            pred: even(),
            body: vec![Stmt::Barrier],
        }]);
        let names: Vec<String> = reductions(&k).into_iter().map(|(n, _)| n).collect();
        assert!(names.contains(&"delete@0".to_string()));
        assert!(names.contains(&"unwrap_guard@0".to_string()));
        assert!(names.contains(&"drop_rem@0".to_string()));
    }

    #[test]
    fn dropping_the_last_helper_level_turns_calls_into_barriers() {
        // Otherwise the reduced kernel would call a function that no longer
        // exists, and every candidate after it would fail to compile.
        let k = Kernel::with_helper(
            vec![Stmt::If {
                pred: even(),
                body: vec![Stmt::CallHelper],
            }],
            1,
        );
        let (_, reduced) = reductions(&k)
            .into_iter()
            .find(|(n, _)| n == "shallower_helper")
            .expect("the helper can be flattened");
        assert_eq!(reduced.helper_depth, 0);
        assert!(reduced.extra_items().is_empty());
        assert!(reduced.render_body().contains("thread::sync_threads();"));
        assert!(!reduced.render_body().contains("barrier_helper"));
    }

    #[test]
    fn the_default_property_pins_the_oracle_as_well_as_the_signature() {
        // A reduction that turns an unsafe kernel safe would preserve
        // "RC001/warning" while destroying the finding, so the oracle is part of
        // what must not move.
        let unsafe_k = Kernel::new(vec![Stmt::If {
            pred: even(),
            body: vec![Stmt::Barrier],
        }]);
        let safe_k = Kernel::new(vec![Stmt::Barrier]);
        let sem_unsafe = interpret(&unsafe_k, Launch::one_block(32));
        let sem_safe = interpret(&safe_k, Launch::one_block(32));

        let analysis = AnalyzerRecord {
            tool: "reconverge".into(),
            version: "0.1.6".into(),
            command: vec![],
            exit_code: Some(0),
            findings: vec![crate::records::Finding {
                code: "RC001".into(),
                confidence: crate::records::Confidence::Warning,
                message: String::new(),
                kernel: None,
                notes: vec![],
            }],
            raw_stdout: String::new(),
            raw_stderr: String::new(),
            witnesses: vec![],
            crashed: false,
            timed_out: false,
        };

        let p = Property::of(&sem_unsafe, &analysis);
        assert!(p.holds(&sem_unsafe, &analysis));
        assert!(
            !p.holds(&sem_safe, &analysis),
            "the same analyzer answer on a now-safe kernel is not the same finding"
        );
        assert!(
            Property::Signature {
                signature: analysis.signature()
            }
            .holds(&sem_safe, &analysis),
            "the signature-only property deliberately does not care"
        );
    }

    #[test]
    fn the_signature_distinguishes_a_warning_with_a_witness_from_one_without() {
        let base = || AnalyzerRecord {
            tool: "reconverge".into(),
            version: "0.1.6".into(),
            command: vec![],
            exit_code: Some(0),
            findings: vec![crate::records::Finding {
                code: "RC001".into(),
                confidence: crate::records::Confidence::Warning,
                message: String::new(),
                kernel: None,
                notes: vec![],
            }],
            raw_stdout: String::new(),
            raw_stderr: String::new(),
            witnesses: vec![],
            crashed: false,
            timed_out: false,
        };
        let without = base();
        let mut with = base();
        with.witnesses = vec!["{}".to_string()];
        assert_ne!(without.signature(), with.signature());
        assert_eq!(without.signature(), "rc001/warning|0w");
    }
}
