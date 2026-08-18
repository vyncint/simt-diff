//! `simt-diff` — the laboratory CLI.
//!
//! Subcommands are independent where possible (brief §25): `generate` writes
//! a case, `analyze` adds the static record, `compare` reads whatever records
//! exist and classifies. Nothing recomputes an earlier stage silently.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use simt_diff::analyzer::{Analyzer, ReconvergeAnalyzer};
use simt_diff::classify::{Evidence, classify};
use simt_diff::emit::{DeviceDep, emit};
use simt_diff::records::{AnalyzerRecord, GeneratorRecord, GpuRunRecord, Launch, SanitizerRecord};
use simt_diff::templates;

const USAGE: &str = "\
simt-diff -- differential laboratory for SIMT static analyzers

Usage:
  simt-diff doctor                        report which stages this host can run
  simt-diff templates                     list the semantic templates
  simt-diff mutate [OPTS]                 enumerate generated kernels with the
                                          oracle and prediction computed for each
  simt-diff generate <template> [OPTS]    write a case directory
  simt-diff analyze <case-dir>            run the analyzer, record findings
  simt-diff ingest <case-dir> [OPTS]      record a GPU run performed elsewhere
  simt-diff compare <case-dir>            classify from the records present
  simt-diff conformance [OPTS]            generate every case, analyze it, check
                                          each prediction and classify each result
  simt-diff show <case-dir>               print the case's verdict
  simt-diff minimize <case-dir> [OPTS]    shrink a case while keeping what makes
                                          it interesting
  simt-diff package <case-dir> [OPTS]     write a standalone reproducer directory
  simt-diff corpus add <case-dir> [OPTS]  record a case as a regression entry
  simt-diff corpus check [OPTS]           rebuild every entry and compare hashes,
                                          without running the analyzer
  simt-diff regress [OPTS]                rebuild every corpus entry and report
                                          generator or analyzer drift

generate options:
  --out <DIR>       parent directory for cases (default: ./cases)
  --seed <N>        generator seed (default: 0)
  --block <N>       block size in threads, repeatable (default: 32)
  --device-path <D> the RUNNER crate takes path deps rooted at D (required
                    when it is built inside a cuda-oxide checkout). The kernel
                    crate always keeps the pinned revision so it analyzes on
                    any host.

ingest options (the GPU host is usually not this host):
  --stdout <FILE>   runner output containing the BLOCK=/VALUES= lines
  --block <N>       block size the run used (default: read from BLOCK=)
  --seconds <F>     wall time of the run
  --watchdog <N>    watchdog the run was given
  --outcome <K>     completed | watchdog-fired | nonzero-exit | ...
  --sanitizer <FILE>  compute-sanitizer output to record alongside it
  --provenance <S>  where this evidence came from (log path, host id)

mutate options:
  --depth <N>       mutation steps from the seeds (default: 1)
  --limit <N>       keep a reproducible subset of this size (default: all)
  --seed <N>        sampling seed (default: 0)
  --block <N>       block size the oracle is computed for (default: 32)
  --seed-template <ID>  restrict to mutants descended from this seed
  --source <ID>     print one generated kernel instead of the table

conformance options:
  --mutants         analyze the generated corpus instead of the hand-written
                    templates; the seeds are kept as depth-0 controls
  --only <SUBSTR>   analyze only cases whose id contains SUBSTR, for running one
                    targeted experiment instead of a sweep
  --depth/--limit/--seed/--block   as for `mutate`

minimize options:
  --out <DIR>       where candidates and MINIMIZED.md go (default: <case>/minimized)
  --signature-only  preserve only the analyzer's answer, not the oracle. Reduces
                    further, and will happily turn the case into a different
                    program that reads the same.
  --reconverge <PATH>

package options:
  --out <DIR>       reproducer directory (default: <case>/reproducer)
  --finding <TEXT>  the one-line claim the package is evidence of

corpus / regress options:
  --corpus <DIR>    corpus directory (default: ./corpus)
  --name <ID>       entry name (corpus add; default: the case's template id)
  --finding <TEXT>  what the entry is evidence of (corpus add)
  --note <TEXT>     why the expected signature is what it is (corpus add)
  --reconverge <PATH>

analyze options:
  --reconverge <PATH>   cargo-reconverge binary (or $SIMT_DIFF_RECONVERGE)

Exit codes: 0 ok, 1 a case classified as interesting, 2 tool error.
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("simt-diff: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Result<u8, String> {
    let Some(cmd) = args.first().map(String::as_str) else {
        print!("{USAGE}");
        return Ok(2);
    };
    match cmd {
        "-h" | "--help" | "help" => {
            print!("{USAGE}");
            Ok(0)
        }
        "doctor" => doctor(),
        "templates" => {
            for t in templates::TEMPLATES {
                println!("{:<32} {:?}", t.id, t.oracle);
                println!("    {}", t.oracle_reason);
            }
            Ok(0)
        }
        "generate" => generate(&args[1..]),
        "mutate" => mutate_cmd(&args[1..]),
        "analyze" => analyze(&args[1..]),
        "ingest" => ingest(&args[1..]),
        "conformance" => conformance(&args[1..]),
        "compare" | "show" => compare(&args[1..]),
        "minimize" => minimize_cmd(&args[1..]),
        "package" => package_cmd(&args[1..]),
        "corpus" => corpus_cmd(&args[1..]),
        "regress" => regress_cmd(&args[1..]),
        other => Err(format!("unknown command `{other}`; try --help")),
    }
}

// ------------------------------------------------------------------ doctor ---

fn doctor() -> Result<u8, String> {
    let mut missing_gpu = false;
    println!("simt-diff {}", env!("CARGO_PKG_VERSION"));
    println!();

    println!("static stage (needs no GPU)");
    report("rustc", probe(&["rustc", "--version"]));
    report("cargo", probe(&["cargo", "--version"]));
    let reconverge = locate_reconverge();
    match &reconverge {
        Some(p) => report("cargo-reconverge", Some(p.display().to_string())),
        None => report("cargo-reconverge", None),
    }

    println!();
    println!("dynamic stage (needs an NVIDIA GPU)");
    for (label, argv) in [
        (
            "nvidia-smi",
            vec![
                "nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
        ),
        ("nvcc", vec!["nvcc", "--version"]),
        ("compute-sanitizer", vec!["compute-sanitizer", "--version"]),
        ("cargo-oxide", vec!["cargo-oxide", "--version"]),
    ] {
        let found = probe(&argv);
        if found.is_none() {
            missing_gpu = true;
        }
        report(label, found);
    }

    println!();
    if missing_gpu {
        println!(
            "This host can generate, analyze and classify. Execution and\n\
             sanitizer stages need a GPU host; `generate` still emits the\n\
             runner crate for one."
        );
    } else {
        println!("Every stage is available on this host.");
    }
    Ok(0)
}

fn probe(argv: &[&str]) -> Option<String> {
    let out = std::process::Command::new(argv[0])
        .args(&argv[1..])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Some(text.lines().next().unwrap_or("").trim().to_string())
}

fn report(label: &str, found: Option<String>) {
    match found {
        Some(v) => println!("  ok      {label:<20} {v}"),
        None => println!("  missing {label:<20} -"),
    }
}

fn locate_reconverge() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SIMT_DIFF_RECONVERGE") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let out = std::process::Command::new("which")
        .arg("cargo-reconverge")
        .output()
        .ok()?;
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!path.is_empty()).then(|| PathBuf::from(path))
}

// ---------------------------------------------------------------- generate ---

fn generate(args: &[String]) -> Result<u8, String> {
    let template_id = args.first().ok_or("generate needs a template id")?;
    let template = templates::find(template_id)
        .ok_or_else(|| format!("unknown template `{template_id}`; try `simt-diff templates`"))?;

    let mut out = PathBuf::from("cases");
    let mut seed = 0u64;
    let mut blocks: Vec<u32> = Vec::new();
    let mut runner_dep = DeviceDep::default();
    let mut i = 1;
    while i < args.len() {
        let value = |i: usize| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--out" => {
                out = PathBuf::from(value(i)?);
                i += 2;
            }
            "--seed" => {
                seed = value(i)?
                    .parse()
                    .map_err(|_| "--seed must be a number".to_string())?;
                i += 2;
            }
            "--device-path" => {
                runner_dep = DeviceDep::Path {
                    crates_dir: value(i)?,
                };
                i += 2;
            }
            "--block" => {
                blocks.push(
                    value(i)?
                        .parse()
                        .map_err(|_| "--block must be a number".to_string())?,
                );
                i += 2;
            }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }
    if blocks.is_empty() {
        blocks.push(32);
    }

    let launches = blocks
        .iter()
        .map(|b| simt_diff::records::Launch::one_block(*b))
        .collect();
    let record = template.record(seed, launches);
    let id = templates::case_id(&record);
    let root = out.join(&id);
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    emit(&root, &record, &DeviceDep::default(), &runner_dep).map_err(|e| e.to_string())?;
    write_json(&root.join("generator.json"), &record)?;

    println!("{}", root.display());
    Ok(0)
}

// ----------------------------------------------------------------- analyze ---

fn analyze(args: &[String]) -> Result<u8, String> {
    let root = PathBuf::from(args.first().ok_or("analyze needs a case directory")?);
    let mut cli = locate_reconverge();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--reconverge" => {
                cli = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }
    let cli = cli.ok_or(
        "cargo-reconverge not found; pass --reconverge <PATH> or set \
         SIMT_DIFF_RECONVERGE",
    )?;

    let analyzer = ReconvergeAnalyzer::new(cli);
    let record = analyzer
        .analyze(&root.join("kernel"))
        .map_err(|e| format!("running the analyzer: {e}"))?;
    write_json(&root.join("analyzer.json"), &record)?;

    println!(
        "{} finding(s) (exit {:?}), {} witness artifact(s)",
        record.findings.len(),
        record.exit_code,
        record.witnesses.len()
    );
    for f in &record.findings {
        println!(
            "  {:<6} {:<9} {}",
            f.code,
            format!("{:?}", f.confidence),
            f.message
        );
    }
    Ok(0)
}

// ------------------------------------------------------------------ mutate ---

/// How a corpus was asked for: the same options serve `mutate` and
/// `conformance --mutants`, so a case seen in one is the same case in the other.
struct CorpusSpec {
    depth: usize,
    limit: Option<usize>,
    seed: u64,
    block: u32,
    seed_template: Option<String>,
}

impl Default for CorpusSpec {
    fn default() -> Self {
        CorpusSpec {
            depth: 1,
            limit: None,
            seed: 0,
            block: 32,
            seed_template: None,
        }
    }
}

fn parse_corpus_args(args: &[String], spec: &mut CorpusSpec) -> Result<Vec<String>, String> {
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let value = |i: usize| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        let num = |i: usize| -> Result<u64, String> {
            value(i)?
                .parse()
                .map_err(|_| format!("{} needs a number", args[i]))
        };
        match args[i].as_str() {
            "--depth" => {
                spec.depth = num(i)? as usize;
                i += 2
            }
            "--limit" => {
                spec.limit = Some(num(i)? as usize);
                i += 2
            }
            "--seed" => {
                spec.seed = num(i)?;
                i += 2
            }
            "--block" => {
                spec.block = num(i)? as u32;
                i += 2
            }
            "--seed-template" => {
                spec.seed_template = Some(value(i)?);
                i += 2
            }
            other => {
                rest.push(other.to_string());
                i += 1
            }
        }
    }
    Ok(rest)
}

/// Build the corpus, and say out loud what was left out.
///
/// A sweep that silently drops cases reads as "covered everything" when it did
/// not, so the sampling is reported on stderr every time it bites.
fn build_corpus(spec: &CorpusSpec) -> Vec<simt_diff::mutate::Mutant> {
    let launch = Launch::one_block(spec.block);
    let mut all = simt_diff::mutate::enumerate(spec.depth, launch);
    if let Some(seed_template) = &spec.seed_template {
        all.retain(|m| &m.seed == seed_template);
    }
    let total = all.len();
    let Some(limit) = spec.limit else { return all };
    let kept = simt_diff::mutate::sample(all, limit, spec.seed);
    if kept.len() < total {
        eprintln!(
            "note: sampled {} of {total} case(s) with seed {}; {} not analyzed",
            kept.len(),
            spec.seed,
            total - kept.len()
        );
    }
    if kept.len() > limit {
        eprintln!(
            "note: --limit {limit} is below the {} control case(s), which are \
             never dropped; kept {}",
            kept.iter().filter(|m| m.depth() == 0).count(),
            kept.len()
        );
    }
    kept
}

fn mutate_cmd(args: &[String]) -> Result<u8, String> {
    let mut spec = CorpusSpec::default();
    let mut source: Option<String> = None;
    let mut i = 0;
    let mut passthrough = Vec::new();
    while i < args.len() {
        match args[i].as_str() {
            "--source" => {
                source = Some(args.get(i + 1).cloned().ok_or("--source needs a case id")?);
                i += 2;
            }
            other => {
                passthrough.push(other.to_string());
                i += 1;
            }
        }
    }
    let leftover = parse_corpus_args(&passthrough, &mut spec)?;
    if let Some(unknown) = leftover.first() {
        return Err(format!("unrecognized argument `{unknown}`"));
    }

    let launch = Launch::one_block(spec.block);
    let corpus = build_corpus(&spec);

    if let Some(id) = source {
        let m = corpus
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| format!("no generated case `{id}` in this corpus"))?;
        print!(
            "{}",
            simt_diff::mutate::record(m, spec.seed, vec![launch]).kernel_source
        );
        return Ok(0);
    }

    println!(
        "{:<54} {:<17} {:<20} {:<12} rule",
        "case", "oracle", "predicted", "basis"
    );
    println!("{}", "-".repeat(126));
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for m in &corpus {
        let sem = simt_diff::interpret::interpret(&m.kernel, launch);
        let p = simt_diff::model::predict(&sem);
        *counts.entry(p.provenance.label()).or_default() += 1;
        println!(
            "{:<54} {:<17} {:<20} {:<12} {}",
            elide(&m.id, 54),
            format!("{:?}", sem.oracle),
            describe_expected(&p.expected),
            p.provenance.label(),
            p.rule
        );
    }
    println!();
    println!(
        "{} case(s) at depth <= {}: {} quoted from the documentation, {} \
         extrapolated from it, {} measured by this laboratory",
        corpus.len(),
        spec.depth,
        counts.get("quoted").copied().unwrap_or(0),
        counts.get("extrapolated").copied().unwrap_or(0),
        counts.get("measured").copied().unwrap_or(0),
    );
    println!(
        "A violated *extrapolated* prediction is a finding about this model. A \
         violated quoted or measured one is a finding about the analyzer."
    );
    Ok(0)
}

fn elide(s: &str, width: usize) -> String {
    if s.len() <= width {
        s.to_string()
    } else {
        format!("{}~", &s[..width - 1])
    }
}

fn describe_expected(e: &simt_diff::prediction::ExpectedStatic) -> String {
    match e {
        simt_diff::prediction::ExpectedStatic::Gating { code } => format!("{code} gating"),
        simt_diff::prediction::ExpectedStatic::WarningOnly { code } => {
            format!("{code} warning-only")
        }
        simt_diff::prediction::ExpectedStatic::Silent => "silent".to_string(),
        simt_diff::prediction::ExpectedStatic::Unspecified => "-".to_string(),
    }
}

// ------------------------------------------------------------- conformance ---

/// Generate every template, analyze it, and report prediction vs behaviour.
///
/// This is the capability report the brief's §33 asks for: the output is a
/// characterization of where the analyzer's documented boundaries actually
/// lie, and every row is either "as documented" or a place where the
/// documentation and the tool disagree.
fn conformance(args: &[String]) -> Result<u8, String> {
    let mut out = PathBuf::from("cases");
    let mut cli = locate_reconverge();
    let mut spec = CorpusSpec::default();
    let mut mutants = false;
    let mut only: Option<String> = None;
    let mut passthrough = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let value = |i: usize| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--out" => {
                out = PathBuf::from(value(i)?);
                i += 2
            }
            "--reconverge" => {
                cli = Some(PathBuf::from(value(i)?));
                i += 2
            }
            "--mutants" => {
                mutants = true;
                i += 1
            }
            "--only" => {
                only = Some(value(i)?);
                i += 2
            }
            other => {
                passthrough.push(other.to_string());
                i += 1
            }
        }
    }
    let leftover = parse_corpus_args(&passthrough, &mut spec)?;
    if let Some(unknown) = leftover.first() {
        return Err(format!("unrecognized argument `{unknown}`"));
    }
    let cli = cli.ok_or("cargo-reconverge not found; set SIMT_DIFF_RECONVERGE")?;
    let analyzer = ReconvergeAnalyzer::new(cli);
    let launch = Launch::one_block(spec.block);

    // The hand-written templates and the generated corpus produce the same kind
    // of record, so one loop serves both and the two are directly comparable.
    let mut records: Vec<GeneratorRecord> = if mutants {
        build_corpus(&spec)
            .iter()
            .map(|m| simt_diff::mutate::record(m, spec.seed, vec![launch]))
            .collect()
    } else {
        templates::TEMPLATES
            .iter()
            .map(|t| t.record(0, vec![launch]))
            .collect()
    };
    if let Some(substr) = &only {
        let before = records.len();
        records.retain(|r| r.template_id.contains(substr.as_str()));
        eprintln!(
            "note: --only {substr} kept {} of {before} case(s); the rest were not \
             analyzed",
            records.len()
        );
        if records.is_empty() {
            return Err(format!("no case id contains `{substr}`"));
        }
    }

    let started = std::time::Instant::now();
    let mut rows: Vec<Row> = Vec::new();
    for (n, record) in records.iter().enumerate() {
        let id = templates::case_id(record);
        let root = out.join(&id);
        std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        emit(&root, record, &DeviceDep::default(), &DeviceDep::default())
            .map_err(|e| e.to_string())?;
        write_json(&root.join("generator.json"), record)?;

        let analysis = analyzer
            .analyze(&root.join("kernel"))
            .map_err(|e| format!("{}: {e}", record.template_id))?;
        write_json(&root.join("analyzer.json"), &analysis)?;

        let report = simt_diff::prediction::check(&record.expected_static, &analysis);
        write_json(&root.join("prediction.json"), &report)?;

        // No GPU run is needed to see a false positive: a kernel that is valid
        // by construction and carries a gating finding is one already.
        let verdict = classify(&Evidence {
            generator: record,
            analyzer: &analysis,
            runs: &[],
            sanitizer: &[],
        });
        write_json(&root.join("differential.json"), &verdict)?;

        eprintln!(
            "  [{:>3}/{}] {:<54} {:?} / {:?}",
            n + 1,
            records.len(),
            elide(&record.template_id, 54),
            report.outcome,
            verdict.classification
        );
        rows.push(Row {
            record: record.clone(),
            analysis,
            report,
            verdict,
            case: id,
        });
    }

    print_conformance(&rows, started.elapsed(), spec.block);
    write_json(&out.join("conformance.json"), &summarize(&rows))?;

    // Only a violated *quoted* prediction, or a classification that says
    // something about the analyzer, is worth a nonzero exit. An extrapolated
    // prediction failing means this laboratory guessed wrong.
    let actionable = rows
        .iter()
        .filter(|r| r.is_actionable_violation() || r.is_interesting())
        .count();
    Ok(u8::from(actionable > 0))
}

struct Row {
    record: GeneratorRecord,
    analysis: AnalyzerRecord,
    report: simt_diff::prediction::PredictionReport,
    verdict: simt_diff::classify::DifferentialResult,
    case: String,
}

impl Row {
    fn violated(&self) -> bool {
        self.report.outcome == simt_diff::prediction::PredictionOutcome::Violated
    }

    /// `quoted`, `extrapolated`, `measured`, or `hand-declared` for the Stage 3
    /// templates, whose predictions were quoted from the documentation one by
    /// one before this model existed.
    fn provenance(&self) -> &'static str {
        match &self.record.prediction_basis {
            Some(b) => b.provenance.label(),
            None => "hand-declared",
        }
    }

    /// Whether a violation here says something about the analyzer rather than
    /// about this laboratory's guesswork. A violated *measured* rule is the
    /// sharpest of the three: it is a regression against a recorded observation.
    fn is_actionable_violation(&self) -> bool {
        self.violated() && self.provenance() != "extrapolated"
    }

    fn is_interesting(&self) -> bool {
        use simt_diff::classify::Classification::*;
        matches!(
            self.verdict.classification,
            PotentialFalsePositive
                | PotentialFalseNegative
                | ConstructionOracleConflict
                | AnalyzerError
                | AnalyzerTimeout
        )
    }

    fn observed(&self) -> String {
        let mut observed: Vec<String> = self
            .analysis
            .findings
            .iter()
            .filter(|f| simt_diff::classify::CONVERGENCE_CODES.contains(&f.code.as_str()))
            .map(|f| format!("{}/{:?}", f.code, f.confidence))
            .collect();
        if !self.analysis.witnesses.is_empty() {
            observed.push(format!("+{}w", self.analysis.witnesses.len()));
        }
        if observed.is_empty() {
            observed.push("silent".to_string());
        }
        observed.join(",")
    }
}

fn print_conformance(rows: &[Row], elapsed: std::time::Duration, block: u32) {
    println!();
    println!(
        "{:<50} {:<17} {:<19} {:<21} {:<10} classification",
        "case", "oracle", "predicted", "observed", "prediction"
    );
    println!("{}", "-".repeat(146));
    for r in rows {
        println!(
            "{:<50} {:<17} {:<19} {:<21} {:<10} {:?}",
            elide(&r.record.template_id, 50),
            format!("{:?}", r.record.oracle),
            describe_expected(&r.record.expected_static),
            r.observed(),
            format!("{:?}", r.report.outcome),
            r.verdict.classification
        );
    }

    let violated: Vec<&Row> = rows.iter().filter(|r| r.violated()).collect();
    let actionable: Vec<&&Row> = violated
        .iter()
        .filter(|r| r.is_actionable_violation())
        .collect();
    let interesting: Vec<&Row> = rows.iter().filter(|r| r.is_interesting()).collect();

    println!();
    println!(
        "{} case(s) analyzed at block={block} in {:.0}s",
        rows.len(),
        elapsed.as_secs_f64()
    );
    println!(
        "  prediction: {} held, {} violated -- {} of those are about the \
         analyzer (a quoted or measured rule) and {} are about this model (an \
         extrapolated one)",
        rows.len() - violated.len(),
        violated.len(),
        actionable.len(),
        violated.len() - actionable.len()
    );
    println!(
        "  classification: {} case(s) worth a human's attention",
        interesting.len()
    );

    for r in &violated {
        println!();
        println!(
            "VIOLATION/{}  {}  ({})",
            r.provenance(),
            r.record.template_id,
            r.case
        );
        println!("    {}", r.report.detail);
        if let Some(basis) = &r.record.prediction_basis {
            println!("    rule: {}", basis.rule);
            println!("    basis: {}", basis.provenance.source());
            match r.provenance() {
                "extrapolated" => println!(
                    "    NOTE: this prediction was inferred, not quoted. The \
                     model is the first suspect, not the analyzer."
                ),
                "measured" => println!(
                    "    NOTE: this rule was written from a recorded run, so a \
                     violation is a regression against that observation."
                ),
                _ => {}
            }
        }
    }

    for r in &interesting {
        println!();
        println!(
            "{:?}  {}  ({})",
            r.verdict.classification, r.record.template_id, r.case
        );
        for line in &r.verdict.observed {
            println!("    observed: {line}");
        }
        for line in &r.verdict.interpretation {
            println!("    reading:  {line}");
        }
        for line in &r.verdict.not_claimed {
            println!("    not claimed: {line}");
        }
    }
}

fn summarize(rows: &[Row]) -> serde_json::Value {
    serde_json::json!({
        "schema": "simt-diff.conformance.v1",
        "cases": rows.iter().map(|r| serde_json::json!({
            "id": r.record.template_id,
            "case": r.case,
            "oracle": r.record.oracle,
            "oracle_reason": r.record.oracle_reason,
            "expected_static": r.record.expected_static,
            "prediction_basis": r.record.prediction_basis,
            "observed": r.observed(),
            "prediction_outcome": r.report.outcome,
            "prediction_detail": r.report.detail,
            "classification": r.verdict.classification,
        })).collect::<Vec<_>>(),
        "totals": {
            "cases": rows.len(),
            "violations": rows.iter().filter(|r| r.violated()).count(),
            "actionable_violations": rows.iter().filter(|r| r.is_actionable_violation()).count(),
            "interesting": rows.iter().filter(|r| r.is_interesting()).count(),
        }
    })
}

// ---------------------------------------------------------------- minimize ---

/// Read the two records a later stage needs from a case directory.
fn load_case(root: &Path) -> Result<(GeneratorRecord, AnalyzerRecord), String> {
    let read = |name: &str| -> Result<String, String> {
        std::fs::read_to_string(root.join(name))
            .map_err(|e| format!("{}: {e}", root.join(name).display()))
    };
    let generator: GeneratorRecord = serde_json::from_str(&read("generator.json")?)
        .map_err(|e| format!("generator.json: {e}"))?;
    let analyzer: AnalyzerRecord = serde_json::from_str(&read("analyzer.json")?)
        .map_err(|e| format!("analyzer.json: {e}; run `simt-diff analyze` first"))?;
    Ok((generator, analyzer))
}

/// Rebuild the IR for a case from the recipe in its record.
///
/// The kernel source is stored, but source is not what a minimizer reduces --
/// the IR is. Rebuilding from the lineage and checking the hash is also how a
/// generator change gets caught here rather than silently minimizing a different
/// program.
fn kernel_of(record: &GeneratorRecord) -> Result<simt_diff::ir::Kernel, String> {
    let basis = record
        .prediction_basis
        .as_ref()
        .ok_or("this case has no generator recipe; only generated cases can be minimized")?;
    let launch = record
        .launches
        .first()
        .copied()
        .unwrap_or(Launch::one_block(32));
    let entry = simt_diff::corpus::CorpusEntry {
        schema: simt_diff::corpus::SCHEMA.to_string(),
        name: record.template_id.clone(),
        finding: String::new(),
        template_id: record.template_id.clone(),
        seed_template: basis.seed_template.clone(),
        lineage: basis.mutation_lineage.clone(),
        launch,
        oracle: record.oracle,
        kernel_sha256: record.kernel_sha256.clone(),
        expected_signature: String::new(),
        analyzer_version: String::new(),
        measured_on: String::new(),
        note: String::new(),
    };
    let kernel = simt_diff::corpus::regenerate(&entry)?;
    let rebuilt = simt_diff::mutate::record_from_recipe(
        &record.template_id,
        &basis.seed_template,
        &basis.mutation_lineage,
        &kernel,
        launch,
    );
    if rebuilt.kernel_sha256 != record.kernel_sha256 {
        return Err(format!(
            "the recipe for `{}` no longer rebuilds the same kernel; the generator \
             changed since this case was written",
            record.template_id
        ));
    }
    Ok(kernel)
}

fn minimize_cmd(args: &[String]) -> Result<u8, String> {
    let root = PathBuf::from(args.first().ok_or("minimize needs a case directory")?);
    let mut out: Option<PathBuf> = None;
    let mut signature_only = false;
    let mut cli = locate_reconverge();
    let mut i = 1;
    while i < args.len() {
        let value = |i: usize| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--out" => {
                out = Some(PathBuf::from(value(i)?));
                i += 2
            }
            "--signature-only" => {
                signature_only = true;
                i += 1
            }
            "--reconverge" => {
                cli = Some(PathBuf::from(value(i)?));
                i += 2
            }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }
    let cli = cli.ok_or("cargo-reconverge not found; set SIMT_DIFF_RECONVERGE")?;
    let (record, analysis) = load_case(&root)?;
    let kernel = kernel_of(&record)?;
    let launch = record
        .launches
        .first()
        .copied()
        .unwrap_or(Launch::one_block(32));
    let sem = simt_diff::interpret::interpret(&kernel, launch);

    let property = if signature_only {
        simt_diff::minimize::Property::Signature {
            signature: analysis.signature(),
        }
    } else {
        simt_diff::minimize::Property::of(&sem, &analysis)
    };
    let workdir = out.unwrap_or_else(|| root.join("minimized"));
    std::fs::create_dir_all(&workdir).map_err(|e| e.to_string())?;

    eprintln!("preserving: {}", property.describe());
    let analyzer = ReconvergeAnalyzer::new(cli);
    let minimizer = simt_diff::minimize::Minimizer {
        analyzer: &analyzer,
        workdir: workdir.clone(),
        launch,
        property: property.clone(),
        device_dep: DeviceDep::default(),
    };
    let outcome = minimizer.run(&kernel)?;
    simt_diff::minimize::write_report(&workdir, &outcome, &property).map_err(|e| e.to_string())?;

    println!(
        "{} -> {} nodes in {} step(s), {} analyzer run(s)",
        outcome.start_size,
        outcome.end_size,
        outcome.accepted.len(),
        outcome.analyses
    );
    for a in &outcome.accepted {
        println!("  applied {a}");
    }
    println!();
    println!("{}", outcome.kernel.render_body());
    println!();
    println!("{}", workdir.join("MINIMIZED.md").display());
    Ok(0)
}

// ----------------------------------------------------------------- package ---

fn package_cmd(args: &[String]) -> Result<u8, String> {
    let root = PathBuf::from(args.first().ok_or("package needs a case directory")?);
    let mut out: Option<PathBuf> = None;
    let mut finding: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        let value = |i: usize| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--out" => {
                out = Some(PathBuf::from(value(i)?));
                i += 2
            }
            "--finding" => {
                finding = Some(value(i)?);
                i += 2
            }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }
    let (record, analysis) = load_case(&root)?;
    let verdict = classify(&Evidence {
        generator: &record,
        analyzer: &analysis,
        runs: &[],
        sanitizer: &[],
    });
    let dir = out.unwrap_or_else(|| root.join("reproducer"));
    let finding = finding.unwrap_or_else(|| {
        format!(
            "{}: expected {:?}, observed {}",
            record.template_id,
            record.expected_static,
            analysis.signature()
        )
    });
    simt_diff::package::write(
        &dir,
        &simt_diff::package::Packaging {
            record: &record,
            analysis: &analysis,
            verdict: &verdict,
            finding: &finding,
            toolchain: toolchain_pin(),
        },
    )
    .map_err(|e| e.to_string())?;
    println!("{}", dir.display());
    println!("  ./verify.sh exits nonzero if the observation moves");
    Ok(0)
}

/// The pin from rust-toolchain.toml, so a package says where it was measured.
fn toolchain_pin() -> String {
    let text = std::fs::read_to_string("rust-toolchain.toml").unwrap_or_default();
    for line in text.lines() {
        if let Some(rest) = line.trim().strip_prefix("channel") {
            return rest
                .trim_start_matches([' ', '='])
                .trim()
                .trim_matches('"')
                .to_string();
        }
    }
    "unknown".to_string()
}

// ------------------------------------------------------------------ corpus ---

fn corpus_cmd(args: &[String]) -> Result<u8, String> {
    match args.first().map(String::as_str) {
        Some("add") => corpus_add(&args[1..]),
        Some("check") => corpus_check(&args[1..]),
        Some(other) => Err(format!(
            "unknown corpus subcommand `{other}`; try `add` or `check`"
        )),
        None => Err("corpus needs a subcommand: add or check".to_string()),
    }
}

/// Generator drift only: rebuild each entry from its recipe and compare the
/// kernel hash.
///
/// Split out from `regress` so it can gate CI. Analyzer drift needs the
/// analyzer, which needs the pinned nightly and a built `cargo-reconverge`;
/// generator drift is this repository's own invariant and costs a second.
fn corpus_check(args: &[String]) -> Result<u8, String> {
    let mut dir = PathBuf::from("corpus");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus" => {
                dir = PathBuf::from(args.get(i + 1).cloned().ok_or("--corpus needs a value")?);
                i += 2;
            }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }
    let entries = simt_diff::corpus::load_dir(&dir)?;
    if entries.is_empty() {
        return Err(format!("no entries in {}", dir.display()));
    }

    let mut drifted = 0usize;
    for entry in &entries {
        match simt_diff::corpus::regenerate(entry) {
            Err(e) => {
                println!("BROKEN          {}: {e}", entry.name);
                drifted += 1;
            }
            Ok(kernel) => {
                let record = simt_diff::mutate::record_from_recipe(
                    &entry.template_id,
                    &entry.seed_template,
                    &entry.lineage,
                    &kernel,
                    entry.launch,
                );
                if record.kernel_sha256 == entry.kernel_sha256 {
                    println!("ok              {}", entry.name);
                } else {
                    println!(
                        "GENERATOR DRIFT {}: recipe now builds {}… , entry records {}…",
                        entry.name,
                        &record.kernel_sha256[..12],
                        &entry.kernel_sha256[..12]
                    );
                    drifted += 1;
                }
            }
        }
    }
    println!();
    println!(
        "{} entry(ies), {drifted} rebuilt differently",
        entries.len()
    );
    if drifted > 0 {
        println!(
            "A corpus entry that no longer rebuilds is not evidence of anything. \
             Either the operator change was intended -- re-record the entry with \
             `corpus add` and say so -- or it was not."
        );
    }
    Ok(u8::from(drifted > 0))
}

fn corpus_add(args: &[String]) -> Result<u8, String> {
    let root = PathBuf::from(args.first().ok_or("corpus add needs a case directory")?);
    let mut dir = PathBuf::from("corpus");
    let mut name: Option<String> = None;
    let mut finding: Option<String> = None;
    let mut note = String::new();
    let mut i = 1;
    while i < args.len() {
        let value = |i: usize| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--corpus" => {
                dir = PathBuf::from(value(i)?);
                i += 2
            }
            "--name" => {
                name = Some(value(i)?);
                i += 2
            }
            "--finding" => {
                finding = Some(value(i)?);
                i += 2
            }
            "--note" => {
                note = value(i)?;
                i += 2
            }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }
    let (record, analysis) = load_case(&root)?;
    let basis = record
        .prediction_basis
        .as_ref()
        .ok_or("only generated cases carry the recipe a corpus entry needs")?;
    // Rebuild before recording, so an entry is never written that cannot be
    // reproduced from its own recipe.
    kernel_of(&record)?;

    let entry = simt_diff::corpus::CorpusEntry {
        schema: simt_diff::corpus::SCHEMA.to_string(),
        name: name.unwrap_or_else(|| slug(&record.template_id)),
        finding: finding.unwrap_or_else(|| record.template_id.clone()),
        template_id: record.template_id.clone(),
        seed_template: basis.seed_template.clone(),
        lineage: basis.mutation_lineage.clone(),
        launch: record
            .launches
            .first()
            .copied()
            .unwrap_or(Launch::one_block(32)),
        oracle: record.oracle,
        kernel_sha256: record.kernel_sha256.clone(),
        expected_signature: analysis.signature(),
        analyzer_version: analysis.version.clone(),
        measured_on: today(),
        note,
    };
    let path = simt_diff::corpus::write(&dir, &entry).map_err(|e| e.to_string())?;
    println!("{}", path.display());
    println!("  signature {}", entry.expected_signature);
    Ok(0)
}

fn slug(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

/// UTC date, for the `measured_on` field. A recorded observation without a date
/// is not a record of anything.
fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86_400;
    let (mut y, mut d) = (1970i64, days as i64);
    loop {
        let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        let len = if leap { 366 } else { 365 };
        if d < len {
            break;
        }
        d -= len;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let months = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 0;
    while d >= months[m] {
        d -= months[m];
        m += 1;
    }
    format!("{y:04}-{:02}-{:02}", m + 1, d + 1)
}

fn regress_cmd(args: &[String]) -> Result<u8, String> {
    let mut dir = PathBuf::from("corpus");
    let mut work = PathBuf::from("cases-regress");
    let mut cli = locate_reconverge();
    let mut i = 0;
    while i < args.len() {
        let value = |i: usize| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--corpus" => {
                dir = PathBuf::from(value(i)?);
                i += 2
            }
            "--out" => {
                work = PathBuf::from(value(i)?);
                i += 2
            }
            "--reconverge" => {
                cli = Some(PathBuf::from(value(i)?));
                i += 2
            }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }
    let cli = cli.ok_or("cargo-reconverge not found; set SIMT_DIFF_RECONVERGE")?;
    let analyzer = ReconvergeAnalyzer::new(cli);
    let entries = simt_diff::corpus::load_dir(&dir)?;
    if entries.is_empty() {
        return Err(format!("no entries in {}", dir.display()));
    }

    let mut results = Vec::new();
    for entry in &entries {
        let (drift, detail) = regress_one(&analyzer, entry, &work);
        eprintln!("  {:<44} {}", entry.name, drift.label());
        results.push((entry, drift, detail));
    }

    println!();
    println!("{:<44} {:<16} detail", "entry", "state");
    println!("{}", "-".repeat(110));
    for (entry, drift, detail) in &results {
        println!("{:<44} {:<16} {detail}", entry.name, drift.label());
    }

    let moved: Vec<_> = results
        .iter()
        .filter(|(_, d, _)| *d != simt_diff::corpus::Drift::None)
        .collect();
    println!();
    println!(
        "{} entry(ies): {} unchanged, {} moved",
        results.len(),
        results.len() - moved.len(),
        moved.len()
    );
    for (entry, drift, detail) in &moved {
        println!();
        println!("{}  {}", drift.label(), entry.name);
        println!("    finding: {}", entry.finding);
        println!(
            "    measured {} with analyzer {}",
            entry.measured_on, entry.analyzer_version
        );
        println!("    {detail}");
        if !entry.note.is_empty() {
            println!("    note: {}", entry.note);
        }
    }
    Ok(u8::from(!moved.is_empty()))
}

fn regress_one(
    analyzer: &ReconvergeAnalyzer,
    entry: &simt_diff::corpus::CorpusEntry,
    work: &Path,
) -> (simt_diff::corpus::Drift, String) {
    use simt_diff::corpus::Drift;

    let kernel = match simt_diff::corpus::regenerate(entry) {
        Ok(k) => k,
        Err(e) => return (Drift::Broken, e),
    };
    let record = simt_diff::mutate::record_from_recipe(
        &entry.template_id,
        &entry.seed_template,
        &entry.lineage,
        &kernel,
        entry.launch,
    );
    if record.kernel_sha256 != entry.kernel_sha256 {
        return (
            Drift::Generator,
            format!(
                "the recipe now builds a different kernel ({}… vs {}…)",
                &record.kernel_sha256[..12],
                &entry.kernel_sha256[..12]
            ),
        );
    }
    let dir = work.join(&entry.name);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return (Drift::Broken, e.to_string());
    }
    if let Err(e) = emit(&dir, &record, &DeviceDep::default(), &DeviceDep::default()) {
        return (Drift::Broken, e.to_string());
    }
    let analysis = match analyzer.analyze(&dir.join("kernel")) {
        Ok(a) => a,
        Err(e) => return (Drift::Broken, e.to_string()),
    };
    let signature = analysis.signature();
    if signature == entry.expected_signature {
        (Drift::None, format!("signature {signature}"))
    } else {
        (
            Drift::Analyzer,
            format!("expected {}, got {signature}", entry.expected_signature),
        )
    }
}

// ------------------------------------------------------------------ ingest ---

/// Record a run that happened on another host.
///
/// The execution stage needs a GPU and the analysis stage does not, so the
/// two routinely happen on different machines. Ingestion is therefore a
/// first-class step rather than hand-edited JSON: the parsing is code, and
/// the provenance of the evidence is recorded with it.
fn ingest(args: &[String]) -> Result<u8, String> {
    let root = PathBuf::from(args.first().ok_or("ingest needs a case directory")?);
    let mut stdout_path: Option<PathBuf> = None;
    let mut sanitizer_path: Option<PathBuf> = None;
    let mut block: Option<u32> = None;
    let mut seconds = 0.0f64;
    let mut watchdog = 20u64;
    let mut outcome = "completed".to_string();
    let mut provenance = String::new();

    let mut i = 1;
    while i < args.len() {
        let value = |i: usize| -> Result<String, String> {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--stdout" => {
                stdout_path = Some(PathBuf::from(value(i)?));
                i += 2
            }
            "--sanitizer" => {
                sanitizer_path = Some(PathBuf::from(value(i)?));
                i += 2
            }
            "--block" => {
                block = Some(value(i)?.parse().map_err(|_| "--block")?);
                i += 2
            }
            "--seconds" => {
                seconds = value(i)?.parse().map_err(|_| "--seconds")?;
                i += 2
            }
            "--watchdog" => {
                watchdog = value(i)?.parse().map_err(|_| "--watchdog")?;
                i += 2
            }
            "--outcome" => {
                outcome = value(i)?;
                i += 2
            }
            "--provenance" => {
                provenance = value(i)?;
                i += 2
            }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }

    let text = match &stdout_path {
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        None => return Err("ingest needs --stdout".into()),
    };
    let (parsed_block, observed) = simt_diff::runner_output::parse(&text)?;
    let block = block
        .or(parsed_block)
        .ok_or("no BLOCK= line in the output and no --block given")?;

    let outcome = simt_diff::runner_output::outcome_from_str(&outcome)?;
    let record = GpuRunRecord {
        launch: simt_diff::records::Launch::one_block(block),
        command: vec![format!("<external: {provenance}>")],
        outcome,
        exit_code: Some(0),
        seconds,
        watchdog_seconds: watchdog,
        stdout: text,
        stderr: String::new(),
        observed,
    };
    let mut runs: Vec<GpuRunRecord> = read_json(&root.join("gpu.json")).unwrap_or_default();
    runs.retain(|r| r.launch != record.launch);
    runs.push(record);
    write_json(&root.join("gpu.json"), &runs)?;

    if let Some(p) = sanitizer_path {
        let raw = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let (reported, errors) = simt_diff::runner_output::parse_sanitizer(&raw);
        let record = SanitizerRecord {
            tool: "synccheck".to_string(),
            command: vec![format!("<external: {provenance}>")],
            launch: simt_diff::records::Launch::one_block(block),
            reported,
            error_count: errors,
            outcome: simt_diff::records::RunOutcome::Completed,
            raw,
        };
        let mut all: Vec<SanitizerRecord> =
            read_json(&root.join("sanitizer.json")).unwrap_or_default();
        all.retain(|s| s.launch != record.launch);
        all.push(record);
        write_json(&root.join("sanitizer.json"), &all)?;
    }

    println!("ingested block={block} into {}", root.display());
    Ok(0)
}

// ----------------------------------------------------------------- compare ---

fn compare(args: &[String]) -> Result<u8, String> {
    let root = PathBuf::from(args.first().ok_or("compare needs a case directory")?);
    let generator: GeneratorRecord = read_json(&root.join("generator.json"))?;
    let analyzer: AnalyzerRecord = read_json(&root.join("analyzer.json"))
        .map_err(|e| format!("{e} -- run `simt-diff analyze` first"))?;
    let runs: Vec<GpuRunRecord> = read_json(&root.join("gpu.json")).unwrap_or_default();
    let sanitizer: Vec<SanitizerRecord> =
        read_json(&root.join("sanitizer.json")).unwrap_or_default();

    let result = classify(&Evidence {
        generator: &generator,
        analyzer: &analyzer,
        runs: &runs,
        sanitizer: &sanitizer,
    });
    write_json(&root.join("differential.json"), &result)?;

    println!("case         {}", root.display());
    println!("template     {}", generator.template_id);
    println!("oracle       {:?}", generator.oracle);
    println!("class        {:?}", result.classification);
    println!("evidence     {:?}", result.strengths);
    println!("\nobserved");
    for o in &result.observed {
        println!("  - {o}");
    }
    println!("\ninterpretation");
    for o in &result.interpretation {
        println!("  - {o}");
    }
    if !result.not_claimed.is_empty() {
        println!("\nnot claimed");
        for o in &result.not_claimed {
            println!("  - {o}");
        }
    }

    let interesting = matches!(
        result.classification,
        simt_diff::classify::Classification::PotentialFalseNegative
            | simt_diff::classify::Classification::PotentialFalsePositive
            | simt_diff::classify::Classification::ConstructionOracleConflict
    );
    Ok(u8::from(interesting))
}

// -------------------------------------------------------------------- json ---

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, format!("{text}\n")).map_err(|e| format!("{}: {e}", path.display()))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}
