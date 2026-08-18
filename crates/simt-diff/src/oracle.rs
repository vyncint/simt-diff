//! What the *generator* knows about a case, independent of any analyzer.
//!
//! This is evidence source A of `docs/research-baseline.md` §5, and the
//! probe in §9.5 is the reason it is first-class rather than a convenience:
//! an invalid warp mask produced a value byte-identical to the valid case,
//! `synccheck` reported nothing, and the analyzer's own corpus records the
//! class at 0% recall. Construction knowledge was the only thing that knew
//! the program was wrong.
//!
//! A `ConstructionOracle` must never be derived from analyzer output. It is
//! set by the template that built the case, from how it was built.

use serde::{Deserialize, Serialize};

/// The semantic class a template asserts about the kernel it generated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ConstructionOracle {
    /// Every thread of the scope reaches every barrier; every collective is
    /// called at a convergent point with a mask naming exactly the
    /// participants.
    KnownSafe,
    /// A barrier is reachable under thread-divergent control by
    /// construction.
    KnownUnsafe,
    /// A collective's supplied mask names lanes that do not participate, or
    /// omits lanes that do.
    KnownMaskInvalid,
    /// A collective's supplied mask names exactly the participating lanes.
    KnownMaskValid,
    /// The case is only well-defined if some value is uniform, and the
    /// template does not establish that.
    KnownRequiresUniformity,
    /// Deliberately outside the analyzer's documented surface
    /// (baseline §2.3), so a missing finding is correct behaviour.
    KnownOutsideAnalyzerScope,
    /// A mutation invalidated the label and no new one could be proven.
    /// Never guess -- see the brief's §11.
    NoOracle,
}

impl ConstructionOracle {
    /// Whether construction alone asserts the program is invalid.
    ///
    /// `KnownRequiresUniformity` is deliberately not invalid: it is a
    /// precondition the template declines to establish, not a bug.
    pub fn asserts_invalid(self) -> bool {
        matches!(self, Self::KnownUnsafe | Self::KnownMaskInvalid)
    }

    /// Whether construction alone asserts the program is valid.
    pub fn asserts_valid(self) -> bool {
        matches!(self, Self::KnownSafe | Self::KnownMaskValid)
    }
}

/// How strongly a case's property is established, per evidence source.
///
/// Deliberately *not* ordered as a scalar: the brief's §30 is explicit that
/// runtime evidence is not "stronger" than static reasoning, because they
/// answer different questions. A case carries a set of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleStrength {
    None,
    /// The generator built it this way.
    Construction,
    /// Every point of the generated finite launch/input domain was run.
    ExhaustiveFiniteDomain,
    /// Observed on hardware, under one launch and one input.
    DynamicObserved,
    /// A vendor dynamic checker reported it.
    SanitizerObserved,
    /// A machine-checkable proof accompanies the case.
    StaticProof,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_two_invalid_classes_assert_invalidity() {
        assert!(ConstructionOracle::KnownUnsafe.asserts_invalid());
        assert!(ConstructionOracle::KnownMaskInvalid.asserts_invalid());
        for ok in [
            ConstructionOracle::KnownSafe,
            ConstructionOracle::KnownMaskValid,
            ConstructionOracle::KnownRequiresUniformity,
            ConstructionOracle::KnownOutsideAnalyzerScope,
            ConstructionOracle::NoOracle,
        ] {
            assert!(!ok.asserts_invalid(), "{ok:?} must not assert invalidity");
        }
    }

    #[test]
    fn requires_uniformity_asserts_neither_way() {
        let o = ConstructionOracle::KnownRequiresUniformity;
        assert!(!o.asserts_invalid());
        assert!(!o.asserts_valid());
    }

    #[test]
    fn oracle_names_are_stable_on_the_wire() {
        let j = serde_json::to_string(&ConstructionOracle::KnownMaskInvalid).unwrap();
        assert_eq!(j, "\"KNOWN_MASK_INVALID\"");
    }
}
