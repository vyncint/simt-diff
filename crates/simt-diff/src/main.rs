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
use simt_diff::records::{AnalyzerRecord, GeneratorRecord, GpuRunRecord, SanitizerRecord};
use simt_diff::templates;

const USAGE: &str = "\
simt-diff -- differential laboratory for SIMT static analyzers

Usage:
  simt-diff doctor                        report which stages this host can run
  simt-diff templates                     list the semantic templates
  simt-diff generate <template> [OPTS]    write a case directory
  simt-diff analyze <case-dir>            run the analyzer, record findings
  simt-diff ingest <case-dir> [OPTS]      record a GPU run performed elsewhere
  simt-diff compare <case-dir>            classify from the records present
  simt-diff conformance [OPTS]            generate every template, analyze it,
                                          and check each documented prediction
  simt-diff show <case-dir>               print the case's verdict

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
        "analyze" => analyze(&args[1..]),
        "ingest" => ingest(&args[1..]),
        "conformance" => conformance(&args[1..]),
        "compare" | "show" => compare(&args[1..]),
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
        ("nvidia-smi", vec!["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"]),
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
    let out = std::process::Command::new(argv[0]).args(&argv[1..]).output().ok()?;
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
    let out = std::process::Command::new("which").arg("cargo-reconverge").output().ok()?;
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
            args.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--out" => {
                out = PathBuf::from(value(i)?);
                i += 2;
            }
            "--seed" => {
                seed = value(i)?.parse().map_err(|_| "--seed must be a number".to_string())?;
                i += 2;
            }
            "--device-path" => {
                runner_dep = DeviceDep::Path { crates_dir: value(i)? };
                i += 2;
            }
            "--block" => {
                blocks.push(value(i)?.parse().map_err(|_| "--block must be a number".to_string())?);
                i += 2;
            }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }
    if blocks.is_empty() {
        blocks.push(32);
    }

    let launches = blocks.iter().map(|b| simt_diff::records::Launch::one_block(*b)).collect();
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
        println!("  {:<6} {:<9} {}", f.code, format!("{:?}", f.confidence), f.message);
    }
    Ok(0)
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
    let mut i = 0;
    while i < args.len() {
        let value = |i: usize| -> Result<String, String> {
            args.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--out" => { out = PathBuf::from(value(i)?); i += 2 }
            "--reconverge" => { cli = Some(PathBuf::from(value(i)?)); i += 2 }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }
    let cli = cli.ok_or("cargo-reconverge not found; set SIMT_DIFF_RECONVERGE")?;
    let analyzer = ReconvergeAnalyzer::new(cli);

    let mut violations = 0usize;
    let mut rows = Vec::new();
    for template in templates::TEMPLATES {
        let record = template.record(0, vec![simt_diff::records::Launch::one_block(32)]);
        let id = templates::case_id(&record);
        let root = out.join(&id);
        std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        emit(&root, &record, &DeviceDep::default(), &DeviceDep::default())
            .map_err(|e| e.to_string())?;
        write_json(&root.join("generator.json"), &record)?;

        let analysis = analyzer
            .analyze(&root.join("kernel"))
            .map_err(|e| format!("{}: {e}", template.id))?;
        write_json(&root.join("analyzer.json"), &analysis)?;

        let report = simt_diff::prediction::check(&record.expected_static, &analysis);
        write_json(&root.join("prediction.json"), &report)?;
        if report.outcome == simt_diff::prediction::PredictionOutcome::Violated {
            violations += 1;
        }
        eprintln!("  {:<34} {:?}", template.id, report.outcome);
        rows.push((template.id, record.expected_static.clone(), analysis, report, id));
    }

    println!();
    println!("{:<34} {:<22} {:<26} {}", "template", "predicted", "observed", "prediction");
    println!("{}", "-".repeat(104));
    for (id, expected, analysis, report, _) in &rows {
        let predicted = match expected {
            simt_diff::prediction::ExpectedStatic::Gating { code } => format!("{code} gating"),
            simt_diff::prediction::ExpectedStatic::WarningOnly { code } => format!("{code} warning-only"),
            simt_diff::prediction::ExpectedStatic::Silent => "silent".to_string(),
            simt_diff::prediction::ExpectedStatic::Unspecified => "-".to_string(),
        };
        let mut observed: Vec<String> = analysis
            .findings
            .iter()
            .filter(|f| simt_diff::classify::CONVERGENCE_CODES.contains(&f.code.as_str()))
            .map(|f| format!("{}/{:?}", f.code, f.confidence))
            .collect();
        if !analysis.witnesses.is_empty() {
            observed.push(format!("+{}w", analysis.witnesses.len()));
        }
        if observed.is_empty() {
            observed.push("silent".to_string());
        }
        println!(
            "{:<34} {:<22} {:<26} {:?}",
            id,
            predicted,
            observed.join(","),
            report.outcome
        );
    }
    println!();
    for (id, _, _, report, case) in &rows {
        if report.outcome == simt_diff::prediction::PredictionOutcome::Violated {
            println!("VIOLATION  {id}  ({case})");
            println!("           {}", report.detail);
        }
    }
    println!(
        "\n{} template(s), {} prediction violation(s)",
        rows.len(),
        violations
    );
    Ok(u8::from(violations > 0))
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
            args.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--stdout" => { stdout_path = Some(PathBuf::from(value(i)?)); i += 2 }
            "--sanitizer" => { sanitizer_path = Some(PathBuf::from(value(i)?)); i += 2 }
            "--block" => { block = Some(value(i)?.parse().map_err(|_| "--block")?); i += 2 }
            "--seconds" => { seconds = value(i)?.parse().map_err(|_| "--seconds")?; i += 2 }
            "--watchdog" => { watchdog = value(i)?.parse().map_err(|_| "--watchdog")?; i += 2 }
            "--outcome" => { outcome = value(i)?; i += 2 }
            "--provenance" => { provenance = value(i)?; i += 2 }
            other => return Err(format!("unrecognized argument `{other}`")),
        }
    }

    let text = match &stdout_path {
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        None => return Err("ingest needs --stdout".into()),
    };
    let (parsed_block, observed) = simt_diff::runner_output::parse(&text)?;
    let block = block.or(parsed_block).ok_or(
        "no BLOCK= line in the output and no --block given",
    )?;

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
    let sanitizer: Vec<SanitizerRecord> = read_json(&root.join("sanitizer.json")).unwrap_or_default();

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
