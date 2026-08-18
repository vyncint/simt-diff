//! Parsing the generated runner's output.
//!
//! Kept separate and tested because this is where hardware evidence enters
//! the record, and a parser that silently returns "no values" would turn a
//! real discrepancy into `DynamicInconclusive` (baseline §9.5 shows values
//! are the only channel that sees an invalid mask).

use std::collections::BTreeMap;

/// What a runner's stdout carried: the block size it reported, and the per-lane
/// values, either of which may be absent.
pub type RunnerOutput = (Option<u32>, Option<BTreeMap<u32, u32>>);

use crate::records::RunOutcome;

/// Returns the block size, if the runner printed one, and the per-lane values.
pub fn parse(text: &str) -> Result<RunnerOutput, String> {
    let mut block = None;
    let mut values = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("BLOCK=") {
            block = rest.trim().parse().ok();
        } else if let Some(rest) = line.strip_prefix("VALUES=") {
            let mut map = BTreeMap::new();
            for (lane, field) in rest.split(',').enumerate() {
                let field = field.trim();
                if field.is_empty() {
                    continue;
                }
                let v: u32 = field
                    .parse()
                    .map_err(|_| format!("lane {lane}: `{field}` is not a u32"))?;
                map.insert(lane as u32, v);
            }
            if map.is_empty() {
                return Err("VALUES= line carried no values".to_string());
            }
            values = Some(map);
        }
    }
    Ok((block, values))
}

pub fn outcome_from_str(s: &str) -> Result<RunOutcome, String> {
    Ok(match s {
        "completed" => RunOutcome::Completed,
        "watchdog-fired" => RunOutcome::WatchdogFired,
        "nonzero-exit" => RunOutcome::NonzeroExit,
        "compile-failed" => RunOutcome::CompileFailed,
        "launch-failed" => RunOutcome::LaunchFailed,
        "not-run" => RunOutcome::NotRun,
        other => return Err(format!("unknown outcome `{other}`")),
    })
}

/// `========= ERROR SUMMARY: N errors`. A missing summary is *not* read as
/// "clean": the tool may have failed to start, and inventing a clean result
/// would be inventing evidence.
pub fn parse_sanitizer(raw: &str) -> (bool, Option<u32>) {
    for line in raw.lines() {
        if let Some(rest) = line.split("ERROR SUMMARY:").nth(1) {
            let n: Option<u32> = rest.split_whitespace().next().and_then(|t| t.parse().ok());
            return (n.is_some_and(|n| n > 0), n);
        }
    }
    (false, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_block_and_values() {
        let (block, values) = parse("BLOCK=32\nVALUES=4294967295,4294967295\n").unwrap();
        assert_eq!(block, Some(32));
        let values = values.unwrap();
        assert_eq!(values[&0], 0xffff_ffff);
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn a_malformed_value_is_an_error_not_a_silent_zero() {
        assert!(parse("VALUES=1,oops,3").is_err());
    }

    #[test]
    fn an_empty_values_line_is_an_error() {
        assert!(parse("VALUES=\n").is_err());
    }

    #[test]
    fn output_without_values_yields_none_rather_than_failing() {
        let (block, values) = parse("BLOCK=64\nsomething else\n").unwrap();
        assert_eq!(block, Some(64));
        assert!(values.is_none());
    }

    #[test]
    fn reads_the_sanitizer_error_summary() {
        assert_eq!(
            parse_sanitizer("========= ERROR SUMMARY: 0 errors"),
            (false, Some(0))
        );
        assert_eq!(
            parse_sanitizer("========= ERROR SUMMARY: 3 errors"),
            (true, Some(3))
        );
    }

    #[test]
    fn a_missing_summary_is_unknown_not_clean() {
        let (reported, count) = parse_sanitizer("sanitizer failed to start");
        assert!(!reported);
        assert_eq!(count, None, "absence of a summary must not read as zero errors");
    }
}
