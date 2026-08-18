//! Versioned, machine-readable records. One per evidence source, plus the
//! differential verdict that reads them.
//!
//! Schema rule, matching reconverge's own: additive-only within a major
//! version, and consumers tolerate unknown fields. Raw tool output is kept
//! verbatim alongside the parsed form -- a parser can be wrong, and the
//! artifact has to outlive the parser.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::oracle::ConstructionOracle;

pub const SCHEMA_CASE: &str = "simt-diff.case.v1";
pub const SCHEMA_DIFFERENTIAL: &str = "simt-diff.differential.v1";

/// Launch geometry. A first-class artifact, per the brief's §12.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Launch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl Launch {
    pub fn one_block(block_x: u32) -> Self {
        Launch { grid: (1, 1, 1), block: (block_x, 1, 1), shared_mem_bytes: 0 }
    }

    /// Lanes per warp is fixed at 32 on every NVIDIA target this supports.
    pub fn warps_per_block(&self) -> u32 {
        self.block.0.div_ceil(32) * self.block.1.max(1) * self.block.2.max(1)
    }
}

/// What the generator built, and what it asserts about it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneratorRecord {
    pub template_id: String,
    pub generator_version: String,
    pub seed: u64,
    pub oracle: ConstructionOracle,
    /// Why the oracle holds, in one sentence, written by the template.
    pub oracle_reason: String,
    /// The kernel source analyzed and executed. Both crates embed this
    /// verbatim; `kernel_sha256` proves they are the same program.
    pub kernel_source: String,
    pub kernel_sha256: String,
    pub kernel_name: String,
    pub launches: Vec<Launch>,
    /// Expected per-lane values, when the template can compute them.
    /// Absent means "no reference model", never "anything is fine".
    pub reference_model: Option<ReferenceModel>,
    /// Set when the analyzer's own documentation already places this class
    /// outside its current scope. Its presence turns a would-be
    /// false-negative claim into `AnalyzerUnsupported`, which is the honest
    /// reading and the difference between characterizing a boundary and
    /// filing a bug against a published limitation.
    #[serde(default)]
    pub documented_limitation: Option<String>,
}

/// A template's own answer for what the kernel should produce.
///
/// Baseline §9.5: without this, an invalid mask is invisible -- it returns
/// a value byte-identical to the valid case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceModel {
    pub description: String,
    /// Expected value per lane index, for the launch it was computed for.
    pub expected: BTreeMap<u32, u32>,
    pub launch: Launch,
}

/// Exactly what the analyzer reported. Never infer a finding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalyzerRecord {
    pub tool: String,
    pub version: String,
    pub command: Vec<String>,
    pub exit_code: Option<i32>,
    pub findings: Vec<Finding>,
    /// `findings.v1` documents exactly as printed, one per line.
    pub raw_stdout: String,
    pub raw_stderr: String,
    /// witness.v1 documents found on disk, verbatim.
    pub witnesses: Vec<String>,
    pub crashed: bool,
    pub timed_out: bool,
}

/// A `findings.v1` finding, deserialized permissively.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Finding {
    pub code: String,
    pub confidence: Confidence,
    pub message: String,
    #[serde(default)]
    pub kernel: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// reconverge's confidence ladder. The distinction is load-bearing: only a
/// gating tier can ever be a false positive (baseline §2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    Deny,
    Confirmed,
    Warning,
}

impl Confidence {
    /// Whether the tool asserts this at a tier that gates CI.
    pub fn gates(self) -> bool {
        matches!(self, Self::Deny | Self::Confirmed)
    }
}

impl AnalyzerRecord {
    /// Findings for the given codes at a gating tier.
    pub fn gating(&self, codes: &[&str]) -> Vec<&Finding> {
        self.findings
            .iter()
            .filter(|f| f.confidence.gates() && codes.contains(&f.code.as_str()))
            .collect()
    }

    /// Findings for the given codes at warning tier only.
    pub fn warnings(&self, codes: &[&str]) -> Vec<&Finding> {
        self.findings
            .iter()
            .filter(|f| f.confidence == Confidence::Warning && codes.contains(&f.code.as_str()))
            .collect()
    }
}

/// One execution of one launch, in its own process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuRunRecord {
    pub launch: Launch,
    pub command: Vec<String>,
    pub outcome: RunOutcome,
    pub exit_code: Option<i32>,
    pub seconds: f64,
    pub watchdog_seconds: u64,
    pub stdout: String,
    pub stderr: String,
    /// Values read back, per lane index, when the runner reported them.
    pub observed: Option<BTreeMap<u32, u32>>,
}

/// What happened to the process. Deliberately free of interpretation: the
/// baseline's §9.2 records a run where the watchdog was killing compilation,
/// and the wording is what kept that from becoming 24 fictitious deadlocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RunOutcome {
    Completed,
    /// The watchdog fired. This is not "deadlock"; see §9.2/§9.3.
    WatchdogFired,
    NonzeroExit,
    CompileFailed,
    LaunchFailed,
    NotRun,
}

/// One sanitized execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SanitizerRecord {
    pub tool: String,
    pub command: Vec<String>,
    pub launch: Launch,
    pub reported: bool,
    pub error_count: Option<u32>,
    pub outcome: RunOutcome,
    pub raw: String,
}

/// Environment, with nothing machine-identifying persisted (brief §23).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EnvironmentRecord {
    pub os: String,
    pub gpu_name: Option<String>,
    pub compute_capability: Option<String>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub sanitizer_version: Option<String>,
    pub rustc_version: Option<String>,
    pub cuda_oxide_revision: Option<String>,
    pub reconverge_version: Option<String>,
    pub simt_diff_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_deny_and_confirmed_gate() {
        assert!(Confidence::Deny.gates());
        assert!(Confidence::Confirmed.gates());
        assert!(!Confidence::Warning.gates());
    }

    #[test]
    fn warp_count_rounds_up() {
        assert_eq!(Launch::one_block(32).warps_per_block(), 1);
        assert_eq!(Launch::one_block(33).warps_per_block(), 2);
        assert_eq!(Launch::one_block(128).warps_per_block(), 4);
    }

    #[test]
    fn findings_deserialize_with_unknown_fields_present() {
        // Schemas are additive-only within v1; an unknown field must not
        // break the adapter.
        let f: Finding = serde_json::from_str(
            r#"{"code":"RC002","confidence":"warning","message":"m",
                "span":{"file":"a.rs","line_start":1,"column_start":1,
                        "line_end":1,"column_end":2},
                "explain":"RC002","future_field":42}"#,
        )
        .expect("unknown fields must be tolerated");
        assert_eq!(f.code, "RC002");
        assert_eq!(f.confidence, Confidence::Warning);
    }
}
