//! The regression corpus (brief §34).
//!
//! A finding that lives in a document decays: the analyzer moves, the finding
//! quietly stops being true, and nobody notices until someone re-reads the
//! document and cannot reproduce it. An entry here is the opposite -- a claim
//! with a machine-checkable expiry date.
//!
//! Entries store the *recipe*, not the program: a seed template and the
//! operators applied to it. Regenerating from the recipe and comparing the
//! kernel hash catches the other kind of drift, where the generator changes and
//! a case silently becomes a different case. Both failures are reported
//! separately, because they mean opposite things: analyzer drift is news about
//! reconverge, generator drift is news about this repository.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::ir::Kernel;
use crate::mutate;
use crate::oracle::ConstructionOracle;
use crate::records::Launch;

pub const SCHEMA: &str = "simt-diff.corpus.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusEntry {
    #[serde(default = "schema")]
    pub schema: String,
    /// Filename-safe identifier, and how a regression names itself.
    pub name: String,
    /// What this case is evidence of, in one sentence.
    pub finding: String,
    /// The generated case's own id, `seed+op+op`. Part of the kernel's doc
    /// comment, so a rebuild needs it to reproduce the same bytes.
    #[serde(default)]
    pub template_id: String,
    pub seed_template: String,
    /// The operators applied to the seed, in order. The recipe.
    pub lineage: Vec<String>,
    pub launch: Launch,
    pub oracle: ConstructionOracle,
    /// Of the *rendered kernel file*, so generator drift is detectable.
    pub kernel_sha256: String,
    /// What the analyzer said when this entry was written.
    pub expected_signature: String,
    pub analyzer_version: String,
    /// ISO date the signature was measured. Written by the tool, never guessed.
    pub measured_on: String,
    /// Why the expected signature is what it is, and what a change would mean.
    pub note: String,
}

fn schema() -> String {
    SCHEMA.to_string()
}

/// Rebuild the kernel from its recipe, applying each operator by name.
///
/// Deliberately not "enumerate everything and search": applying the named
/// operators in order is what makes the recipe a recipe, and it fails loudly if
/// an operator this entry depends on no longer exists.
pub fn regenerate(entry: &CorpusEntry) -> Result<Kernel, String> {
    let mut kernel = mutate::seed(&entry.seed_template)
        .ok_or_else(|| format!("seed template `{}` no longer exists", entry.seed_template))?;
    for op in &entry.lineage {
        let candidates = mutate::mutations(&kernel, entry.launch);
        let (_, next) = candidates
            .into_iter()
            .find(|(name, _)| name == op)
            .ok_or_else(|| format!("operator `{op}` no longer applies to this case"))?;
        kernel = next;
    }
    Ok(kernel)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drift {
    /// The recipe rebuilds the same program and the analyzer still says the same
    /// thing.
    None,
    /// The recipe now builds a different program. News about this repository.
    Generator,
    /// The program is the same and the analyzer's answer changed. News about
    /// reconverge -- which is the whole point of keeping the entry.
    Analyzer,
    /// The recipe cannot be rebuilt at all.
    Broken,
}

impl Drift {
    pub fn label(self) -> &'static str {
        match self {
            Drift::None => "ok",
            Drift::Generator => "GENERATOR DRIFT",
            Drift::Analyzer => "ANALYZER DRIFT",
            Drift::Broken => "BROKEN",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RegressResult {
    pub name: String,
    pub drift: Drift,
    pub detail: String,
}

pub fn load_dir(dir: &Path) -> Result<Vec<CorpusEntry>, String> {
    let mut entries = Vec::new();
    let read = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut paths: Vec<_> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    for path in paths {
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let entry: CorpusEntry = serde_json::from_str(&text)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        entries.push(entry);
    }
    Ok(entries)
}

pub fn write(dir: &Path, entry: &CorpusEntry) -> std::io::Result<std::path::PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.json", entry.name));
    let mut text = serde_json::to_string_pretty(entry).expect("entry serializes");
    text.push('\n');
    std::fs::write(&path, text)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(seed: &str, lineage: &[&str]) -> CorpusEntry {
        CorpusEntry {
            schema: schema(),
            name: "t".into(),
            finding: "test".into(),
            template_id: format!("{seed}{}", lineage.iter().map(|l| format!("+{l}")).collect::<String>()),
            seed_template: seed.into(),
            lineage: lineage.iter().map(|s| s.to_string()).collect(),
            launch: Launch::one_block(32),
            oracle: ConstructionOracle::KnownUnsafe,
            kernel_sha256: String::new(),
            expected_signature: String::new(),
            analyzer_version: "0.1.6".into(),
            measured_on: "2026-08-18".into(),
            note: String::new(),
        }
    }

    #[test]
    fn a_recipe_rebuilds_the_same_program_every_time() {
        let e = entry("barrier_divergent_intra_warp", &["complementary_guard@0", "clone_guard_to_end@0.0"]);
        let a = regenerate(&e).unwrap();
        let b = regenerate(&e).unwrap();
        assert_eq!(mutate::fingerprint(&a), mutate::fingerprint(&b));
        // The A, B, A shape this recipe describes.
        let body = a.render_body();
        let guards: Vec<&str> = body
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("if ") && !l.contains("out.get_mut"))
            .collect();
        assert_eq!(guards.len(), 3);
    }

    #[test]
    fn a_missing_seed_or_operator_fails_loudly_instead_of_guessing() {
        let bad_seed = entry("no_such_template", &[]);
        assert!(regenerate(&bad_seed).unwrap_err().contains("no longer exists"));

        let bad_op = entry("barrier_uniform", &["invert_guard@0"]);
        assert!(regenerate(&bad_op).unwrap_err().contains("no longer applies"));
    }

    #[test]
    fn entries_round_trip_through_json_with_unknown_fields_tolerated() {
        let e = entry("barrier_uniform", &[]);
        let mut value: serde_json::Value = serde_json::to_value(&e).unwrap();
        value["something_a_later_version_added"] = serde_json::json!(true);
        let back: CorpusEntry = serde_json::from_value(value).unwrap();
        assert_eq!(back.seed_template, "barrier_uniform");
        assert_eq!(back.schema, SCHEMA);
    }
}
