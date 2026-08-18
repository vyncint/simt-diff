//! A closed intermediate representation for probe kernels.
//!
//! Stage 3 exhausted what hand-written templates can find: fourteen of them,
//! fourteen predictions held. Widening the search means generating kernels
//! nobody thought to write, and the brief's §11 is emphatic that a mutation
//! must never leave the case with a guessed semantic label.
//!
//! This IR is how that promise is kept. It is deliberately small enough that
//! [`crate::interpret`] can execute every program it can express, exactly,
//! over every thread of the launch. So a mutated kernel does not inherit its
//! parent's oracle: the oracle is *recomputed* from the mutant itself.
//!
//! Everything renderable here is a construct already observed to compile and
//! run on sm_86 (`docs/stage-1.md`, `docs/conformance-reconverge-0.1.6.md`),
//! with two deliberate additions -- truncating casts and a two-level helper
//! chain -- which exist precisely because they sit on documented analyzer
//! boundaries.

/// How a value is rendered: Rust infers integer literals in a comparison, but
/// a loop bound compared against a `u32` counter needs the cast the hand
/// templates already carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ctx {
    Inferred,
    U32,
}

/// A per-thread integer the guards are built from.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    /// `i.get()` -- the thread's linear index.
    LaneIndex,
    /// `warp::warp_id()` -- a lane-environment read.
    WarpId,
    /// `warp::lanemask_lt().count_ones()` -- a lane-environment read, equal to
    /// the thread's lane within its warp.
    LaneMaskLtPopcount,
    /// A literal. Only reachable as a loop bound, where it makes the trip count
    /// uniform -- which is the one way to put a divergent guard inside a loop
    /// without the loop itself being divergent.
    Const(u32),
    Rem(Box<Value>, u32),
    Div(Box<Value>, u32),
    BitAnd(Box<Value>, u32),
    /// `(x as u8) as u32`. The README names truncating casts as something the
    /// witness interpreter does not yet have, which is why this is here.
    TruncU8(Box<Value>),
}

impl Value {
    pub fn eval(&self, tid: u32) -> u32 {
        match self {
            Value::LaneIndex => tid,
            Value::WarpId => tid / 32,
            Value::LaneMaskLtPopcount => tid % 32,
            Value::Const(v) => *v,
            Value::Rem(v, k) => {
                if *k == 0 {
                    0
                } else {
                    v.eval(tid) % k
                }
            }
            Value::Div(v, k) => {
                if *k == 0 {
                    0
                } else {
                    v.eval(tid) / k
                }
            }
            Value::BitAnd(v, k) => v.eval(tid) & k,
            Value::TruncU8(v) => v.eval(tid) & 0xff,
        }
    }

    pub fn reads_lane_index(&self) -> bool {
        match self {
            Value::LaneIndex => true,
            Value::WarpId | Value::LaneMaskLtPopcount | Value::Const(_) => false,
            Value::Rem(v, _) | Value::Div(v, _) | Value::BitAnd(v, _) | Value::TruncU8(v) => {
                v.reads_lane_index()
            }
        }
    }

    pub fn reads_lane_environment(&self) -> bool {
        match self {
            Value::WarpId | Value::LaneMaskLtPopcount => true,
            Value::LaneIndex | Value::Const(_) => false,
            Value::Rem(v, _) | Value::Div(v, _) | Value::BitAnd(v, _) | Value::TruncU8(v) => {
                v.reads_lane_environment()
            }
        }
    }

    pub fn has_truncating_cast(&self) -> bool {
        match self {
            Value::TruncU8(_) => true,
            Value::LaneIndex | Value::WarpId | Value::LaneMaskLtPopcount | Value::Const(_) => false,
            Value::Rem(v, _) | Value::Div(v, _) | Value::BitAnd(v, _) => v.has_truncating_cast(),
        }
    }

    pub fn uses_warp_api(&self) -> bool {
        self.reads_lane_environment()
    }

    pub fn render(&self, ctx: Ctx) -> String {
        match self {
            Value::LaneIndex => match ctx {
                Ctx::Inferred => "i.get()".to_string(),
                Ctx::U32 => "i.get() as u32".to_string(),
            },
            Value::WarpId => "warp::warp_id()".to_string(),
            Value::LaneMaskLtPopcount => "warp::lanemask_lt().count_ones()".to_string(),
            Value::Const(v) => v.to_string(),
            Value::Rem(v, k) => format!("{} % {k}", v.render(ctx)),
            Value::Div(v, k) => format!("{} / {k}", v.render(ctx)),
            Value::BitAnd(v, k) => format!("{} & {}", v.render(ctx), hex32(*k)),
            // The inner value is rendered in inferred context: `i.get() as u8`
            // is what the cast has to bite on, not `i.get() as u32 as u8`.
            Value::TruncU8(v) => format!("({} as u8) as u32", v.render(Ctx::Inferred)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    pub fn apply(self, a: u32, b: u32) -> bool {
        match self {
            CmpOp::Eq => a == b,
            CmpOp::Ne => a != b,
            CmpOp::Lt => a < b,
            CmpOp::Le => a <= b,
            CmpOp::Gt => a > b,
            CmpOp::Ge => a >= b,
        }
    }

    /// The exact logical negation, so a flip operator cannot silently change
    /// the shape of the guard as well as its sense.
    pub fn negate(self) -> CmpOp {
        match self {
            CmpOp::Eq => CmpOp::Ne,
            CmpOp::Ne => CmpOp::Eq,
            CmpOp::Lt => CmpOp::Ge,
            CmpOp::Ge => CmpOp::Lt,
            CmpOp::Gt => CmpOp::Le,
            CmpOp::Le => CmpOp::Gt,
        }
    }

    pub fn render(self) -> &'static str {
        match self {
            CmpOp::Eq => "==",
            CmpOp::Ne => "!=",
            CmpOp::Lt => "<",
            CmpOp::Le => "<=",
            CmpOp::Gt => ">",
            CmpOp::Ge => ">=",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Pred {
    Cmp(Value, CmpOp, u32),
    Not(Box<Pred>),
    And(Box<Pred>, Box<Pred>),
}

impl Pred {
    pub fn eval(&self, tid: u32) -> bool {
        match self {
            Pred::Cmp(v, op, rhs) => op.apply(v.eval(tid), *rhs),
            Pred::Not(p) => !p.eval(tid),
            Pred::And(a, b) => a.eval(tid) && b.eval(tid),
        }
    }

    pub fn values(&self) -> Vec<&Value> {
        match self {
            Pred::Cmp(v, _, _) => vec![v],
            Pred::Not(p) => p.values(),
            Pred::And(a, b) => {
                let mut v = a.values();
                v.extend(b.values());
                v
            }
        }
    }

    pub fn reads_lane_index(&self) -> bool {
        self.values().iter().any(|v| v.reads_lane_index())
    }

    pub fn reads_lane_environment(&self) -> bool {
        self.values().iter().any(|v| v.reads_lane_environment())
    }

    pub fn has_truncating_cast(&self) -> bool {
        self.values().iter().any(|v| v.has_truncating_cast())
    }

    pub fn render(&self) -> String {
        match self {
            Pred::Cmp(v, op, rhs) => {
                format!("{} {} {rhs}", v.render(Ctx::Inferred), op.render())
            }
            Pred::Not(p) => format!("!({})", p.render()),
            Pred::And(a, b) => format!("{} && {}", a.render(), b.render()),
        }
    }
}

/// Where a collective's mask comes from. The provenance matters more than the
/// value: reconverge documents that it can only evaluate literals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mask {
    /// A hex literal in the call.
    Literal(u32),
    /// A named `const`, which the analyzer documents it cannot evaluate.
    NamedConst(u32),
    /// `warp::active_mask()` -- names exactly the lanes present.
    ActiveMask,
    /// `warp::ballot(..)`, the unmasked wrapper, which forwards a full mask
    /// from inside cuda-device. Documented as outside the v1 surface.
    ImplicitWrapper,
}

impl Mask {
    /// The lane set the mask names, warp-local. `None` for [`Mask::ActiveMask`],
    /// which names the participants by definition and so cannot be compared
    /// against them.
    pub fn named_lanes(self) -> Option<u32> {
        match self {
            Mask::Literal(v) | Mask::NamedConst(v) => Some(v),
            Mask::ImplicitWrapper => Some(0xffff_ffff),
            Mask::ActiveMask => None,
        }
    }

    pub fn is_literal(self) -> bool {
        matches!(self, Mask::Literal(_))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Stmt {
    /// `thread::sync_threads()`
    Barrier,
    /// `b = warp::ballot_sync(<mask>, true)`
    Ballot {
        mask: Mask,
    },
    If {
        pred: Pred,
        body: Vec<Stmt>,
    },
    /// `while <counter> < (<bound>) { body; <counter> += 1 }`
    Loop {
        bound: Value,
        body: Vec<Stmt>,
    },
    /// `barrier_helper()` -- the barrier is one call away.
    CallHelper,
}

/// A whole probe kernel: the body of `pub fn probe(mut out: DisjointSlice<u32>)`
/// plus the items it needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Kernel {
    pub stmts: Vec<Stmt>,
    /// 0 = no helper; 1 = `barrier_helper`; 2 = it calls
    /// `barrier_helper_inner`. Two levels test whether the summary-based
    /// interprocedural pass propagates transitively.
    pub helper_depth: usize,
}

/// What the kernel writes to `out`, derived from its shape rather than stored,
/// so a mutation cannot leave it inconsistent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteExpr {
    One,
    Ballot,
    LoopCounter,
}

impl Kernel {
    pub fn new(stmts: Vec<Stmt>) -> Self {
        Kernel {
            stmts,
            helper_depth: 0,
        }
    }

    pub fn with_helper(stmts: Vec<Stmt>, helper_depth: usize) -> Self {
        Kernel {
            stmts,
            helper_depth,
        }
    }

    pub fn write_expr(&self) -> WriteExpr {
        if self.count(&|s| matches!(s, Stmt::Ballot { .. })) > 0 {
            WriteExpr::Ballot
        } else if self.count(&|s| matches!(s, Stmt::Loop { .. })) > 0 {
            WriteExpr::LoopCounter
        } else {
            WriteExpr::One
        }
    }

    fn count(&self, pred: &dyn Fn(&Stmt) -> bool) -> usize {
        fn walk(stmts: &[Stmt], pred: &dyn Fn(&Stmt) -> bool) -> usize {
            stmts
                .iter()
                .map(|s| {
                    let here = usize::from(pred(s));
                    here + match s {
                        Stmt::If { body, .. } | Stmt::Loop { body, .. } => walk(body, pred),
                        _ => 0,
                    }
                })
                .sum()
        }
        walk(&self.stmts, pred)
    }

    /// Every statement, with its address, in a fixed pre-order. The address is
    /// how mutation operators and site facts refer to the same place.
    pub fn walk(&self) -> Vec<(Vec<usize>, &Stmt)> {
        fn go<'a>(stmts: &'a [Stmt], prefix: &[usize], out: &mut Vec<(Vec<usize>, &'a Stmt)>) {
            for (i, s) in stmts.iter().enumerate() {
                let mut path = prefix.to_vec();
                path.push(i);
                out.push((path.clone(), s));
                match s {
                    Stmt::If { body, .. } | Stmt::Loop { body, .. } => go(body, &path, out),
                    _ => {}
                }
            }
        }
        let mut out = Vec::new();
        go(&self.stmts, &[], &mut out);
        out
    }

    /// The vector containing `path`'s last component, plus that index. Every
    /// mutation operator goes through this, so none of them can address a
    /// statement that does not exist.
    pub fn locate_mut(&mut self, path: &[usize]) -> Option<(&mut Vec<Stmt>, usize)> {
        let (last, prefix) = path.split_last()?;
        let mut cursor = &mut self.stmts;
        for step in prefix {
            let next = cursor.get_mut(*step)?;
            cursor = match next {
                Stmt::If { body, .. } | Stmt::Loop { body, .. } => body,
                _ => return None,
            };
        }
        if *last >= cursor.len() {
            return None;
        }
        Some((cursor, *last))
    }

    pub fn get(&self, path: &[usize]) -> Option<&Stmt> {
        let mut cursor = &self.stmts;
        let (last, prefix) = path.split_last()?;
        for step in prefix {
            cursor = match cursor.get(*step)? {
                Stmt::If { body, .. } | Stmt::Loop { body, .. } => body,
                _ => return None,
            };
        }
        cursor.get(*last)
    }

    pub fn uses_warp_api(&self) -> bool {
        self.walk().iter().any(|(_, s)| match s {
            Stmt::Ballot { .. } => true,
            Stmt::If { pred, .. } => pred.reads_lane_environment(),
            Stmt::Loop { bound, .. } => bound.uses_warp_api(),
            _ => false,
        })
    }

    pub fn extra_uses(&self) -> Vec<&'static str> {
        if self.uses_warp_api() {
            vec!["warp"]
        } else {
            vec![]
        }
    }

    /// Named-mask consts and helper functions, emitted before the kernel and
    /// identical in the analyzed and the executed crate.
    pub fn extra_items(&self) -> String {
        let mut items = String::new();
        for (_, s) in self.walk() {
            if let Stmt::Ballot {
                mask: Mask::NamedConst(v),
            } = s
            {
                let decl = format!("pub const {}: u32 = {};\n", const_name(*v), hex32(*v));
                if !items.contains(&decl) {
                    items.push_str(&decl);
                }
            }
        }
        if self.helper_depth >= 1 {
            if !items.is_empty() {
                items.push('\n');
            }
            let inner = if self.helper_depth >= 2 {
                items.push_str(
                    "#[inline(never)]\npub fn barrier_helper() {\n    \
                     barrier_helper_inner();\n}\n\n",
                );
                "barrier_helper_inner"
            } else {
                "barrier_helper"
            };
            items.push_str(&format!(
                "#[inline(never)]\npub fn {inner}() {{\n    thread::sync_threads();\n}}\n"
            ));
        }
        items
    }

    /// The rendered function body, indented to match the hand-written
    /// templates so generated and hand cases read identically.
    pub fn render_body(&self) -> String {
        let mut lines: Vec<String> = vec!["let i = thread::index_1d();".to_string()];
        let counters = self.loop_counters();
        for name in &counters {
            lines.push(format!("let mut {name} = 0u32;"));
        }

        let ballots = self.count(&|s| matches!(s, Stmt::Ballot { .. }));
        let inline_ballot = ballots == 1
            && self
                .walk()
                .iter()
                .any(|(p, s)| p.len() == 1 && matches!(s, Stmt::Ballot { .. }));
        if ballots > 0 && !inline_ballot {
            lines.push("let mut b = 0u32;".to_string());
        }

        let mut state = RenderState {
            counters,
            next_counter: 0,
            inline_ballot,
        };
        render_stmts(&self.stmts, 0, &mut state, &mut lines);

        let write = match self.write_expr() {
            WriteExpr::One => "1".to_string(),
            WriteExpr::Ballot => "b".to_string(),
            WriteExpr::LoopCounter => state
                .counters
                .first()
                .cloned()
                .unwrap_or_else(|| "1".to_string()),
        };
        lines.push(format!(
            "if let Some(e) = out.get_mut(i) {{ *e = {write}; }}"
        ));

        lines
            .iter()
            .map(|l| {
                if l.is_empty() {
                    String::new()
                } else {
                    format!("{}{l}", " ".repeat(8))
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Counter names, one per loop, in pre-order. The first is what the kernel
    /// writes out, matching the hand-written loop template.
    pub fn loop_counters(&self) -> Vec<String> {
        let n = self.count(&|s| matches!(s, Stmt::Loop { .. }));
        (0..n)
            .map(|i| {
                if i == 0 {
                    "n".to_string()
                } else {
                    format!("n{i}")
                }
            })
            .collect()
    }
}

struct RenderState {
    counters: Vec<String>,
    next_counter: usize,
    inline_ballot: bool,
}

fn render_stmts(stmts: &[Stmt], depth: usize, state: &mut RenderState, out: &mut Vec<String>) {
    let pad = "    ".repeat(depth);
    for s in stmts {
        match s {
            Stmt::Barrier => out.push(format!("{pad}thread::sync_threads();")),
            Stmt::CallHelper => out.push(format!("{pad}barrier_helper();")),
            Stmt::Ballot { mask } => {
                let call = match mask {
                    Mask::Literal(v) => format!("warp::ballot_sync({}, true)", hex32(*v)),
                    Mask::NamedConst(v) => {
                        format!("warp::ballot_sync({}, true)", const_name(*v))
                    }
                    Mask::ActiveMask => "warp::ballot_sync(warp::active_mask(), true)".to_string(),
                    Mask::ImplicitWrapper => "warp::ballot(true)".to_string(),
                };
                if state.inline_ballot {
                    out.push(format!("{pad}let b = {call};"));
                } else {
                    out.push(format!("{pad}b = {call};"));
                }
            }
            Stmt::If { pred, body } => {
                out.push(format!("{pad}if {} {{", pred.render()));
                render_stmts(body, depth + 1, state, out);
                out.push(format!("{pad}}}"));
            }
            Stmt::Loop { bound, body } => {
                let name = state
                    .counters
                    .get(state.next_counter)
                    .cloned()
                    .unwrap_or_else(|| "n".to_string());
                state.next_counter += 1;
                // A nested loop's counter is declared at the top of the body,
                // so it has to be reset where the loop starts or the second
                // entry would begin where the first stopped -- and then the
                // interpreter and the emitted program would disagree.
                if depth > 0 {
                    out.push(format!("{pad}{name} = 0;"));
                }
                out.push(format!(
                    "{pad}while {name} < ({}) {{",
                    bound.render(Ctx::U32)
                ));
                render_stmts(body, depth + 1, state, out);
                out.push(format!("{}{name} += 1;", "    ".repeat(depth + 1)));
                out.push(format!("{pad}}}"));
            }
        }
    }
}

/// `0xffff_ffff` -- the grouping the hand-written templates use, so a mask
/// literal reads the same whoever wrote it.
pub fn hex32(v: u32) -> String {
    format!("0x{:04x}_{:04x}", v >> 16, v & 0xffff)
}

fn const_name(v: u32) -> &'static str {
    if v == 0xffff_ffff {
        "FULL_MASK"
    } else {
        "PARTIAL_MASK"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn even_guard() -> Pred {
        Pred::Cmp(Value::Rem(Box::new(Value::LaneIndex), 2), CmpOp::Eq, 0)
    }

    #[test]
    fn lane_environment_values_match_their_hardware_meaning() {
        // lanemask_lt().count_ones() is the lane's index within its warp.
        assert_eq!(Value::LaneMaskLtPopcount.eval(0), 0);
        assert_eq!(Value::LaneMaskLtPopcount.eval(31), 31);
        assert_eq!(Value::LaneMaskLtPopcount.eval(33), 1);
        assert_eq!(Value::WarpId.eval(31), 0);
        assert_eq!(Value::WarpId.eval(32), 1);
    }

    #[test]
    fn a_truncating_cast_truncates() {
        let v = Value::TruncU8(Box::new(Value::LaneIndex));
        assert_eq!(v.eval(255), 255);
        assert_eq!(v.eval(256), 0);
        assert_eq!(v.render(Ctx::Inferred), "(i.get() as u8) as u32");
        assert!(v.has_truncating_cast());
    }

    #[test]
    fn negation_is_exact_and_involutive() {
        for op in [
            CmpOp::Eq,
            CmpOp::Ne,
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Gt,
            CmpOp::Ge,
        ] {
            assert_eq!(op.negate().negate(), op);
            for a in 0..8u32 {
                for b in 0..8u32 {
                    assert_ne!(op.apply(a, b), op.negate().apply(a, b));
                }
            }
        }
    }

    #[test]
    fn the_divergent_barrier_renders_the_hand_written_body() {
        // The hand-written template in templates.rs is the reference: if the
        // renderer cannot reproduce it exactly, the IR is not describing the
        // same programs the measured conformance rows came from.
        let k = Kernel::new(vec![Stmt::If {
            pred: even_guard(),
            body: vec![Stmt::Barrier],
        }]);
        assert_eq!(
            k.render_body(),
            "        let i = thread::index_1d();\n\
             \x20       if i.get() % 2 == 0 {\n\
             \x20           thread::sync_threads();\n\
             \x20       }\n\
             \x20       if let Some(e) = out.get_mut(i) { *e = 1; }"
        );
    }

    #[test]
    fn the_loop_template_renders_its_counter_the_same_way() {
        let k = Kernel::new(vec![Stmt::Loop {
            bound: Value::Rem(Box::new(Value::LaneIndex), 4),
            body: vec![Stmt::Barrier],
        }]);
        assert_eq!(
            k.render_body(),
            "        let i = thread::index_1d();\n\
             \x20       let mut n = 0u32;\n\
             \x20       while n < (i.get() as u32 % 4) {\n\
             \x20           thread::sync_threads();\n\
             \x20           n += 1;\n\
             \x20       }\n\
             \x20       if let Some(e) = out.get_mut(i) { *e = n; }"
        );
    }

    #[test]
    fn a_single_top_level_ballot_is_a_let_binding_and_a_guarded_one_is_not() {
        let plain = Kernel::new(vec![Stmt::Ballot {
            mask: Mask::Literal(0xffff_ffff),
        }]);
        assert!(
            plain
                .render_body()
                .contains("let b = warp::ballot_sync(0xffff_ffff, true);")
        );

        let guarded = Kernel::new(vec![Stmt::If {
            pred: even_guard(),
            body: vec![Stmt::Ballot {
                mask: Mask::Literal(0xffff_ffff),
            }],
        }]);
        let body = guarded.render_body();
        assert!(body.contains("let mut b = 0u32;"));
        assert!(body.contains("b = warp::ballot_sync(0xffff_ffff, true);"));
    }

    #[test]
    fn two_helper_levels_emit_a_transitive_chain() {
        let k = Kernel::with_helper(
            vec![Stmt::If {
                pred: even_guard(),
                body: vec![Stmt::CallHelper],
            }],
            2,
        );
        let items = k.extra_items();
        assert!(items.contains("pub fn barrier_helper() {\n    barrier_helper_inner();"));
        assert!(items.contains("pub fn barrier_helper_inner() {\n    thread::sync_threads();"));
    }

    #[test]
    fn addresses_locate_the_statement_they_name() {
        let k = Kernel::new(vec![Stmt::If {
            pred: even_guard(),
            body: vec![
                Stmt::Barrier,
                Stmt::Ballot {
                    mask: Mask::ActiveMask,
                },
            ],
        }]);
        assert_eq!(
            k.get(&[0, 1]),
            Some(&Stmt::Ballot {
                mask: Mask::ActiveMask
            })
        );
        assert_eq!(k.get(&[0, 2]), None);
        assert_eq!(k.walk().len(), 3);
    }

    #[test]
    fn mask_grouping_matches_the_hand_written_literals() {
        assert_eq!(hex32(0xffff_ffff), "0xffff_ffff");
        assert_eq!(hex32(0x0000_ffff), "0x0000_ffff");
        assert_eq!(hex32(0x5555_5555), "0x5555_5555");
    }
}
