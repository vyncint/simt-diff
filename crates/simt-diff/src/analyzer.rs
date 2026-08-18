//! The analyzer adapter. Only reconverge exists; the trait is here because
//! the brief asks for it, not to host imaginary implementations (§28).
//!
//! Everything goes through published interfaces (baseline §2.2): the JSONL
//! on stdout, the witness files in `<target>/reconverge/`, and the exit code.
//! No linking against `reconverge-core`, which sits behind a trait boundary
//! its own CI gate protects.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use crate::records::{AnalyzerRecord, Finding};

pub trait Analyzer {
    fn analyze(&self, kernel_crate: &Path) -> io::Result<AnalyzerRecord>;
    fn version(&self) -> io::Result<String>;
}

pub struct ReconvergeAnalyzer {
    /// `cargo-reconverge` binary.
    pub cli: PathBuf,
    /// Prepended to PATH so the CLI finds its matching driver.
    pub driver_dir: Option<PathBuf>,
    pub timeout: Duration,
    /// Include warning-tier findings. Always on here: a laboratory that
    /// cannot see the warning tier cannot tell "declined to promote" from
    /// "did not see it", and that distinction decides whether a case is a
    /// miss or documented behaviour (baseline §2.4).
    pub strict: bool,
}

impl ReconvergeAnalyzer {
    pub fn new(cli: PathBuf) -> Self {
        let driver_dir = cli.parent().map(Path::to_path_buf);
        ReconvergeAnalyzer {
            cli,
            driver_dir,
            timeout: Duration::from_secs(600),
            strict: true,
        }
    }

    fn command(&self, kernel_crate: &Path) -> Command {
        let mut cmd = Command::new(&self.cli);
        // `cargo-reconverge` is invoked by cargo as `cargo reconverge ...`,
        // so argv[1] is the subcommand name when run directly.
        cmd.arg("reconverge")
            .arg("check")
            .arg("--message-format")
            .arg("json");
        if self.strict {
            cmd.arg("--strict");
        }
        cmd.current_dir(kernel_crate);
        if let Some(dir) = &self.driver_dir {
            let path = std::env::var("PATH").unwrap_or_default();
            cmd.env("PATH", format!("{}:{}", dir.display(), path));
        }
        cmd
    }
}

impl Analyzer for ReconvergeAnalyzer {
    fn analyze(&self, kernel_crate: &Path) -> io::Result<AnalyzerRecord> {
        // Witness artifacts are read off disk afterwards, so a previous run's
        // files in the same crate directory would be counted as this run's.
        // `regress` re-analyzes into one working directory by design, and a
        // stale `witness-*.json` there turns "declined to promote" into
        // "promoted" -- silently, and in the direction that hides a regression.
        let _ = std::fs::remove_dir_all(kernel_crate.join("target").join("reconverge"));

        let mut cmd = self.command(kernel_crate);
        let rendered: Vec<String> = std::iter::once(self.cli.display().to_string())
            .chain(cmd.get_args().map(|a| a.to_string_lossy().to_string()))
            .collect();

        let started = Instant::now();
        let output = cmd.output()?;
        let elapsed = started.elapsed();

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit = output.status.code();

        // Exit 2 is the documented "tool error" code. Anything without a code
        // was killed by a signal.
        let crashed = exit.is_none_or(|c| c == 2);

        Ok(AnalyzerRecord {
            tool: "reconverge".to_string(),
            version: self.version().unwrap_or_else(|_| "unknown".to_string()),
            command: rendered,
            exit_code: exit,
            findings: parse_findings(&stdout),
            witnesses: collect_witnesses(kernel_crate),
            raw_stdout: stdout,
            raw_stderr: stderr,
            crashed,
            timed_out: elapsed >= self.timeout,
        })
    }

    fn version(&self) -> io::Result<String> {
        // The CLI has no --version of its own; the pinned crate version is
        // what the artifacts carry, and every findings.v1 document repeats
        // it, so read it from output when available.
        Ok(std::env::var("SIMT_DIFF_RECONVERGE_VERSION").unwrap_or_else(|_| "0.1.6".to_string()))
    }
}

/// `--message-format json` prints one `findings.v1` document per crate, one
/// per line. Lines that are not JSON documents are ignored rather than
/// treated as errors: cargo may interleave its own output.
pub fn parse_findings(stdout: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Ok(doc) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if doc.get("schema").and_then(|s| s.as_str()) != Some("findings.v1") {
            continue;
        }
        let Some(findings) = doc.get("findings").and_then(|f| f.as_array()) else {
            continue;
        };
        for finding in findings {
            match serde_json::from_value::<Finding>(finding.clone()) {
                Ok(f) => out.push(f),
                // A finding we cannot parse is kept visible as a synthetic
                // record rather than silently dropped: schemas are
                // additive-only, but an unparseable finding is still evidence.
                Err(e) => out.push(Finding {
                    code: "PARSE".to_string(),
                    confidence: crate::records::Confidence::Warning,
                    message: format!("unparseable finding: {e}"),
                    kernel: None,
                    notes: vec![finding.to_string()],
                }),
            }
        }
    }
    out
}

/// Witness artifacts land in `<target>/reconverge/witness-*.json` and exist
/// only for findings the interpreter replayed (baseline §2.2).
fn collect_witnesses(kernel_crate: &Path) -> Vec<String> {
    let dir = kernel_crate.join("target").join("reconverge");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let is_witness = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("witness-") && n.ends_with(".json"));
        if is_witness && let Ok(text) = std::fs::read_to_string(&path) {
            out.push(text);
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::Confidence;

    const DOC: &str = r#"{"schema":"findings.v1","tool":{"name":"reconverge","version":"0.1.6"},"crate":"case-kernel","findings":[{"code":"RC002","confidence":"warning","message":"collective","kernel":"probe","span":{"file":"src/lib.rs","line_start":9,"column_start":17,"line_end":9,"column_end":52},"explain":"RC002","notes":["mask not evaluable"]}]}"#;

    #[test]
    fn parses_a_findings_document_from_jsonl() {
        let f = parse_findings(DOC);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].code, "RC002");
        assert_eq!(f[0].confidence, Confidence::Warning);
        assert_eq!(f[0].notes, vec!["mask not evaluable".to_string()]);
    }

    #[test]
    fn ignores_interleaved_non_json_output() {
        let mixed = format!("   Compiling case-kernel v0.0.0\n{DOC}\n    Finished\n");
        assert_eq!(parse_findings(&mixed).len(), 1);
    }

    #[test]
    fn ignores_documents_of_other_schemas() {
        let other = r#"{"schema":"unimap.v1","findings":[{"code":"RC001"}]}"#;
        assert!(parse_findings(other).is_empty());
    }

    #[test]
    fn an_unparseable_finding_is_surfaced_not_dropped() {
        let bad = r#"{"schema":"findings.v1","tool":{"name":"r","version":"0"},"crate":"c","findings":[{"code":"RC001"}]}"#;
        let f = parse_findings(bad);
        assert_eq!(f.len(), 1, "evidence must not vanish");
        assert_eq!(f[0].code, "PARSE");
    }

    #[test]
    fn multiple_crates_accumulate() {
        let two = format!("{DOC}\n{DOC}");
        assert_eq!(parse_findings(&two).len(), 2);
    }
}
