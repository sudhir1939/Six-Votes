"""Run this before you zip your work:  python check_submission.py

It checks that every required file exists and that the two JSON files have the
required keys. It does not check whether your numbers are right.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
problems = []


def need(path: str) -> Path | None:
    p = ROOT / path
    if not p.exists():
        problems.append(f"missing: {path}")
        return None
    return p


def keys(obj: dict, wanted: set, where: str) -> None:
    missing = wanted - set(obj)
    if missing:
        problems.append(f"{where}: missing keys {sorted(missing)}")


p = need("outputs/results.json")
if p:
    r = json.loads(p.read_text())
    keys(r, {"n_questions", "models", "leaderboard", "top2", "rank_stability", "panel", "recommended_panel"}, "results.json")
    keys(r.get("top2", {}), {"leader", "runner_up", "accuracy_gap", "gap_ci_low", "gap_ci_high", "p_value", "significant"}, "results.json top2")
    keys(r.get("panel", {}), {"phi_matrix", "mean_phi", "n_eff", "vote_accuracy", "majority_correct_rate", "majority_correct_rate_if_independent"}, "results.json panel")
    keys(r.get("recommended_panel", {}), {"k", "models", "dev_accuracy", "test_accuracy", "baseline_top_k_models", "baseline_top_k_dev_accuracy", "baseline_top_k_test_accuracy"}, "results.json recommended_panel")

p = need("outputs/subliminal_results.json")
if p:
    s = json.loads(p.read_text())
    keys(s, {"config", "per_seed", "mean", "own_experiment"}, "subliminal_results.json")
    if len(s.get("per_seed", [])) != 5:
        problems.append("subliminal_results.json: per_seed needs 5 entries (seeds 0 to 4)")
    if len(s.get("own_experiment", {}).get("conditions", [])) < 3:
        problems.append("subliminal_results.json: own_experiment needs at least 3 conditions")

need("outputs/REPORT.md")
need("submission/NOTES.md")
need("submission/FINDINGS.md")
if not list((ROOT / "submission").glob("one_pager.*")):
    problems.append("missing: submission/one_pager.pdf or submission/one_pager.docx")

if problems:
    print("Not ready:")
    for line in problems:
        print("  -", line)
    sys.exit(1)
print("All required files are present and well-formed.")
