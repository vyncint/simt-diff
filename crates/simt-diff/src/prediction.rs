//! What a template predicts the analyzer will do, and whether it did.
//!
//! This is the addition that turns the laboratory from "look for
//! disagreements" into something sharper: for the limitations reconverge
//! *documents*, the expected behaviour is specific and checkable. The
//! interprocedural rule is not "may be imprecise", it is "call-site findings
//! stay at `warning` and are never witness-promoted". A case that comes back
//! `confirmed` violates that, and so does one that comes back silent — in
//! opposite directions, both worth knowing.
//!
//! Predictions are declared per template, from the analyzer's documentation.
//! They are not derived from a run, or the check would be vacuous.

use serde::{Deserialize, Serialize};

use crate::records::AnalyzerRecord;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExpectedStatic {
    /// A finding with this code, at a tier that gates.
    Gating { code: String },
    /// This code, at warning tier only — seen, but deliberately not promoted.
    WarningOnly { code: String },
    /// Nothing for any convergence code, and that is correct behaviour.
    Silent,
    /// The template makes no prediction.
    Unspecified,
}

/// The const-constructible form, for `static` template tables. Converted to
/// [`ExpectedStatic`] when a record is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpectedStaticSpec {
    Gating(&'static str),
    WarningOnly(&'static str),
    Silent,
    Unspecified,
}

impl From<ExpectedStaticSpec> for ExpectedStatic {
    fn from(spec: ExpectedStaticSpec) -> Self {
        match spec {
            ExpectedStaticSpec::Gating(code) => {
                ExpectedStatic::Gating { code: code.to_string() }
            }
            ExpectedStaticSpec::WarningOnly(code) => {
                ExpectedStatic::WarningOnly { code: code.to_string() }
            }
            ExpectedStaticSpec::Silent => ExpectedStatic::Silent,
            ExpectedStaticSpec::Unspecified => ExpectedStatic::Unspecified,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionOutcome {
    Held,
    Violated,
    NotChecked,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionReport {
    pub expected: ExpectedStatic,
    pub outcome: PredictionOutcome,
    pub detail: String,
}

/// Compare a template's prediction against what the analyzer actually said.
pub fn check(expected: &ExpectedStatic, analyzer: &AnalyzerRecord) -> PredictionReport {
    let codes = crate::classify::CONVERGENCE_CODES;
    let gating = analyzer.gating(codes);
    let warnings = analyzer.warnings(codes);

    let describe = || {
        let mut parts = Vec::new();
        for f in &gating {
            parts.push(format!("{}/{:?}", f.code, f.confidence));
        }
        for f in &warnings {
            parts.push(format!("{}/{:?}", f.code, f.confidence));
        }
        if parts.is_empty() {
            "no RC001/RC002 finding".to_string()
        } else {
            parts.join(", ")
        }
    };

    let (outcome, detail) = match expected {
        ExpectedStatic::Unspecified => (
            PredictionOutcome::NotChecked,
            "the template makes no prediction".to_string(),
        ),
        ExpectedStatic::Silent => {
            if gating.is_empty() && warnings.is_empty() {
                (PredictionOutcome::Held, "silent, as predicted".to_string())
            } else {
                (
                    PredictionOutcome::Violated,
                    format!("predicted silence, got {}", describe()),
                )
            }
        }
        ExpectedStatic::Gating { code } => {
            if gating.iter().any(|f| &f.code == code) {
                (
                    PredictionOutcome::Held,
                    format!("{code} at a gating tier, as predicted"),
                )
            } else {
                (
                    PredictionOutcome::Violated,
                    format!("predicted {code} at a gating tier, got {}", describe()),
                )
            }
        }
        ExpectedStatic::WarningOnly { code } => {
            let promoted: Vec<_> = gating.iter().filter(|f| &f.code == code).collect();
            let warned = warnings.iter().any(|f| &f.code == code);
            if !promoted.is_empty() {
                (
                    PredictionOutcome::Violated,
                    format!(
                        "predicted {code} at warning tier only, but it was \
                         promoted to {:?} -- the documented rule says this \
                         construct is never witness-promoted",
                        promoted[0].confidence
                    ),
                )
            } else if warned {
                (
                    PredictionOutcome::Held,
                    format!("{code} at warning tier only, as predicted"),
                )
            } else {
                (
                    PredictionOutcome::Violated,
                    format!(
                        "predicted {code} at warning tier, got {} -- the \
                         construct was not seen at all",
                        describe()
                    ),
                )
            }
        }
    };

    // A witness artifact for a warning-only prediction is its own violation,
    // independent of the tier: witness promotion is what the rule forbids.
    if matches!(expected, ExpectedStatic::WarningOnly { .. })
        && !analyzer.witnesses.is_empty()
        && outcome == PredictionOutcome::Held
    {
        return PredictionReport {
            expected: expected.clone(),
            outcome: PredictionOutcome::Violated,
            detail: format!(
                "{detail}, but {} witness artifact(s) exist and the rule is that \
                 this construct is never witness-promoted",
                analyzer.witnesses.len()
            ),
        };
    }

    PredictionReport { expected: expected.clone(), outcome, detail }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::{Confidence, Finding};

    fn record(findings: Vec<(&str, Confidence)>, witnesses: usize) -> AnalyzerRecord {
        AnalyzerRecord {
            tool: "reconverge".into(),
            version: "0.1.6".into(),
            command: vec![],
            exit_code: Some(0),
            findings: findings
                .into_iter()
                .map(|(code, confidence)| Finding {
                    code: code.into(),
                    confidence,
                    message: String::new(),
                    kernel: None,
                    notes: vec![],
                })
                .collect(),
            raw_stdout: String::new(),
            raw_stderr: String::new(),
            witnesses: vec!["{}".to_string(); witnesses],
            crashed: false,
            timed_out: false,
        }
    }

    #[test]
    fn silence_predicted_and_delivered() {
        let r = check(&ExpectedStatic::Silent, &record(vec![], 0));
        assert_eq!(r.outcome, PredictionOutcome::Held);
    }

    #[test]
    fn silence_predicted_but_a_finding_appeared() {
        let r = check(
            &ExpectedStatic::Silent,
            &record(vec![("RC002", Confidence::Warning)], 0),
        );
        assert_eq!(r.outcome, PredictionOutcome::Violated);
    }

    #[test]
    fn warning_only_promoted_to_confirmed_is_a_violation() {
        let r = check(
            &ExpectedStatic::WarningOnly { code: "RC001".into() },
            &record(vec![("RC001", Confidence::Confirmed)], 1),
        );
        assert_eq!(r.outcome, PredictionOutcome::Violated);
        assert!(r.detail.contains("never witness-promoted"));
    }

    #[test]
    fn warning_only_not_seen_at_all_is_also_a_violation() {
        // The opposite direction, and just as informative: the documented
        // rule says the construct is seen and held back, not missed.
        let r = check(&ExpectedStatic::WarningOnly { code: "RC001".into() }, &record(vec![], 0));
        assert_eq!(r.outcome, PredictionOutcome::Violated);
        assert!(r.detail.contains("not seen at all"));
    }

    #[test]
    fn warning_only_with_a_witness_is_a_violation_even_at_warning_tier() {
        let r = check(
            &ExpectedStatic::WarningOnly { code: "RC001".into() },
            &record(vec![("RC001", Confidence::Warning)], 1),
        );
        assert_eq!(r.outcome, PredictionOutcome::Violated);
    }

    #[test]
    fn gating_predicted_and_delivered() {
        let r = check(
            &ExpectedStatic::Gating { code: "RC001".into() },
            &record(vec![("RC001", Confidence::Confirmed)], 1),
        );
        assert_eq!(r.outcome, PredictionOutcome::Held);
    }

    #[test]
    fn gating_predicted_but_only_a_warning_arrived() {
        let r = check(
            &ExpectedStatic::Gating { code: "RC001".into() },
            &record(vec![("RC001", Confidence::Warning)], 0),
        );
        assert_eq!(r.outcome, PredictionOutcome::Violated);
    }
}
