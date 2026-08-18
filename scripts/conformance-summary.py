#!/usr/bin/env python3
"""Group a conformance sweep by model rule, so a 133-row table fits on a page.

Reads the `conformance.json` a sweep writes and prints markdown. The point is
that the numbers in docs/stage-4.md come from the artifact rather than from
anyone's memory of it: regenerate, diff, and the doc is either current or wrong.

    scripts/conformance-summary.py cases-final-d1/conformance.json
"""
import json
import sys
from collections import defaultdict


def main(path: str) -> int:
    doc = json.load(open(path))
    cases = doc["cases"]

    by_rule = defaultdict(lambda: {"held": 0, "violated": 0, "examples": []})
    by_class = defaultdict(int)
    for c in cases:
        basis = c.get("prediction_basis") or {}
        rule = basis.get("rule", "(hand-declared)")
        prov = (basis.get("provenance") or {}).get("kind", "quoted")
        row = by_rule[(rule, prov)]
        if c["prediction_outcome"] == "held":
            row["held"] += 1
        else:
            row["violated"] += 1
            if len(row["examples"]) < 2:
                row["examples"].append(c["id"])
        by_class[c["classification"]] += 1

    print(f"### {len(cases)} cases, grouped by the rule that predicted them\n")
    print("| rule | provenance | held | violated | example violation |")
    print("|---|---|---:|---:|---|")
    for (rule, prov), row in sorted(by_rule.items(), key=lambda kv: (-kv[1]["violated"], kv[0])):
        ex = row["examples"][0] if row["examples"] else ""
        print(f"| `{rule}` | {prov} | {row['held']} | {row['violated']} | {ex} |")

    print("\n### classifications\n")
    print("| classification | cases |")
    print("|---|---:|")
    for k, v in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"| {k} | {v} |")

    # Counted here rather than read from `totals`, so a sweep recorded before the
    # summary logic changed still summarizes correctly: the per-case provenance is
    # the durable fact, the totals are a convenience.
    violated = [c for c in cases if c["prediction_outcome"] != "held"]
    def prov(c):
        b = c.get("prediction_basis") or {}
        return (b.get("provenance") or {}).get("kind", "hand-declared")
    about_analyzer = [c for c in violated if prov(c) != "extrapolated"]
    interesting = [
        c for c in cases
        if c["classification"] in {
            "PotentialFalsePositive", "PotentialFalseNegative",
            "ConstructionOracleConflict", "AnalyzerError", "AnalyzerTimeout",
        }
    ]
    print(
        f"\n**{len(cases)} cases: {len(cases) - len(violated)} predictions held, "
        f"{len(violated)} violated — {len(about_analyzer)} of those about the "
        f"analyzer (a quoted or measured rule), "
        f"{len(violated) - len(about_analyzer)} about this model (an extrapolated "
        f"one). {len(interesting)} case(s) classified as needing a human.**"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "cases/conformance.json"))
