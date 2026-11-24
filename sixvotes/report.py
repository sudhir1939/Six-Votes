"""Part 4 - run the whole audit and write results.json and REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import leaderboard as lb
from . import panel as pn
from .io import Panel


def _build_markdown_report(results: dict) -> str:
    """Render a concise, readable summary for a hiring-product manager."""
    n_q = results["n_questions"]
    models = results["models"]
    lb_rows = results["leaderboard"]
    top2 = results["top2"]
    stability = results["rank_stability"]
    panel_stats = results["panel"]
    rec = results["recommended_panel"]

    lines: list[str] = [
        "# Six Votes Audit Report",
        "",
        f"- **Questions audited:** {n_q}",
        f"- **Models evaluated ({len(models)}):** {', '.join(models)}",
        "",
        "## 1. Leaderboard & Top-Two Gap",
        "",
        "| Rank | Model | Accuracy | 95% CI |",
        "| --- | --- | --- | --- |",
    ]

    for row in lb_rows:
        lines.append(
            f"| {row['rank']} | {row['model']} | {row['accuracy']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
        )

    sig_text = "statistically significant (p < 0.05)" if top2["significant"] else "not statistically significant (p >= 0.05)"
    lines.extend(
        [
            "",
            f"- **Top two:** **{top2['leader']}** vs. **{top2['runner_up']}** "
            f"(gap = {top2['accuracy_gap']:.4f}, 95% CI [{top2['gap_ci_low']:.4f}, {top2['gap_ci_high']:.4f}], "
            f"exact McNemar p = {top2['p_value']:.4f} -> {sig_text}).",
            "",
            "## 2. Rank Stability Across Benchmark Sizes",
            "",
            "| Sample Size (n) | P(Full-Data Leader Holds #1) |",
            "| --- | --- |",
        ]
    )

    for s in stability:
        lines.append(f"| {s['n']} | {s['p_leader_holds']:.4f} |")

    lines.extend(
        [
            "",
            "## 3. Error Correlation & Voting Panel",
            "",
            f"- **Mean off-diagonal error correlation (mean phi):** {panel_stats['mean_phi']:.4f}",
            f"- **Effective independent models (n_eff):** {panel_stats['n_eff']:.2f} of {len(models)}",
            f"- **Plurality vote accuracy:** {panel_stats['vote_accuracy']:.4f}",
            f"- **Majority-correct rate (actual):** {panel_stats['majority_correct_rate']:.4f}",
            f"- **Majority-correct rate (if independent):** {panel_stats['majority_correct_rate_if_independent']:.4f}",
            "",
            "## 4. Recommended Panel (k = {})".format(rec["k"]),
            "",
            f"- **Selected panel:** {', '.join(rec['models'])} "
            f"(dev accuracy: {rec['dev_accuracy']:.4f}, test accuracy: {rec['test_accuracy']:.4f})",
            f"- **Baseline top-k:** {', '.join(rec['baseline_top_k_models'])} "
            f"(dev accuracy: {rec['baseline_top_k_dev_accuracy']:.4f}, "
            f"test accuracy: {rec['baseline_top_k_test_accuracy']:.4f})",
            "",
        ]
    )

    return "\n".join(lines)


def run_audit(panel: Panel, out_dir: str | Path, seed: int = 0, n_boot: int = 2000, k: int = 3) -> dict:
    """Compute everything from Parts 1 and 2 and write two files into out_dir:

        results.json   exact layout in docs/OUTPUT_CONTRACT.md (this file is auto-graded)
        REPORT.md      a short report a hiring-product manager could read (Part 4)

    Must work for ANY predictions file with the same four meta columns: any number
    of models (2 or more), any model names, any answer labels, any number of rows.
    """
    correct = np.asarray(panel.correct, dtype=bool)
    answers = np.asarray(panel.answers)
    gold = np.asarray(panel.gold)
    models = list(panel.models)
    n_questions, n_models = correct.shape

    accs = lb.accuracy(correct)
    ci_low, ci_high = lb.bootstrap_accuracy_ci(correct, n_boot=n_boot, seed=seed)

    # Sort best first; tie-break by original CSV column index for determinism
    ranked_indices = sorted(range(n_models), key=lambda idx: (-float(accs[idx]), idx))
    leaderboard_list = [
        {
            "model": models[idx],
            "accuracy": float(accs[idx]),
            "ci_low": float(ci_low[idx]),
            "ci_high": float(ci_high[idx]),
            "rank": rank_pos + 1,
        }
        for rank_pos, idx in enumerate(ranked_indices)
    ]

    top2 = lb.compare_top_two(correct, models, n_boot=n_boot, seed=seed)
    sizes = lb.default_sizes(n_questions)
    stability = lb.rank_stability(correct, sizes=sizes, n_draws=n_boot, seed=seed)

    phi = pn.phi_matrix(correct)
    mean_phi = pn.mean_offdiag(phi)
    n_eff = pn.n_effective(n_models, mean_phi)
    vote_acc = pn.vote_accuracy(answers, gold)
    maj_actual = pn.majority_correct_rate(correct)
    maj_indep = pn.majority_correct_if_independent(accs)

    effective_k = min(int(k), n_models)
    recommended_panel = pn.select_panel(panel, k=effective_k)

    results = {
        "n_questions": int(n_questions),
        "models": models,
        "leaderboard": leaderboard_list,
        "top2": top2,
        "rank_stability": stability,
        "panel": {
            "phi_matrix": [[float(x) for x in row] for row in phi],
            "mean_phi": float(mean_phi),
            "n_eff": float(n_eff),
            "vote_accuracy": float(vote_acc),
            "majority_correct_rate": float(maj_actual),
            "majority_correct_rate_if_independent": float(maj_indep),
        },
        "recommended_panel": recommended_panel,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with (out_path / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report_md = _build_markdown_report(results)
    with (out_path / "REPORT.md").open("w", encoding="utf-8") as f:
        f.write(report_md)

    return results