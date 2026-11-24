"""Public self-checks. Run `pytest -q` from the project root.

Passing these means your functions agree with small cases worked out by hand.
It does not mean the submission is complete: a private grader checks your JSON
outputs on this dataset and on a second dataset you have not seen.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from sixvotes import leaderboard as lb
from sixvotes import panel as pn

T, F = True, False
# 4 questions, 3 models. Model 0: 3/4, model 1: 2/4, model 2: 2/4.
CORRECT = np.array([[T, T, F], [T, F, T], [T, T, F], [F, F, T]])


def test_accuracy():
    assert np.allclose(lb.accuracy(CORRECT), [0.75, 0.5, 0.5])


def test_bootstrap_ci_brackets_the_accuracy():
    rng = np.random.default_rng(1)
    correct = rng.random((400, 2)) < np.array([0.8, 0.6])
    low, high = lb.bootstrap_accuracy_ci(correct, n_boot=500, seed=0)
    acc = correct.mean(axis=0)
    assert low.shape == high.shape == (2,)
    assert np.all(low < acc) and np.all(acc < high)
    assert np.all(high - low < 0.12) and np.all(high - low > 0.04)


def test_mcnemar_known_values():
    a = np.array([T] * 10 + [F] * 2)
    b = np.array([F] * 10 + [T] * 2)          # 10 vs 2 discordant
    assert lb.mcnemar_exact_p(a, b) == pytest.approx(0.0386, abs=1e-3)
    assert lb.mcnemar_exact_p(a, a) == 1.0     # no discordant questions


def test_compare_top_two_keys():
    out = lb.compare_top_two(CORRECT, ["m0", "m1", "m2"], n_boot=200)
    assert out["leader"] == "m0"
    assert out["accuracy_gap"] == pytest.approx(0.25)
    assert set(out) == {"leader", "runner_up", "accuracy_gap", "gap_ci_low",
                        "gap_ci_high", "p_value", "significant"}


def test_rank_stability_shape():
    rng = np.random.default_rng(2)
    correct = rng.random((300, 3)) < np.array([0.9, 0.5, 0.5])
    out = lb.rank_stability(correct, [20, 300], n_draws=200)
    assert [o["n"] for o in out] == [20, 300]
    assert out[-1]["p_leader_holds"] > 0.99


def test_phi_identical_and_opposite():
    c = np.array([[T, T, F], [F, F, T], [T, T, F], [F, F, T]])
    phi = pn.phi_matrix(c)
    assert phi[0, 1] == pytest.approx(1.0)     # same mistakes
    assert phi[0, 2] == pytest.approx(-1.0)    # opposite mistakes
    assert np.allclose(np.diag(phi), 1.0)


def test_phi_constant_model_is_zero():
    c = np.array([[T, T], [T, F], [T, T], [T, F]])   # model 0 is never wrong
    assert pn.phi_matrix(c)[0, 1] == 0.0


def test_n_effective_limits():
    assert pn.n_effective(6, 0.0) == pytest.approx(6.0)   # independent
    assert pn.n_effective(6, 1.0) == pytest.approx(1.0)   # six copies of one model
    assert pn.n_effective(6, 0.4) == pytest.approx(2.0)


def test_plurality_vote_and_tie_break():
    answers = np.array([["A", "B", "B"],     # B wins
                        ["C", "A", "B"],     # three-way tie -> first column -> C
                        ["A", "B", "A"]])    # A wins
    assert list(pn.plurality_vote(answers)) == ["B", "C", "A"]
    assert pn.vote_accuracy(answers, np.array(["B", "C", "B"])) == pytest.approx(2 / 3)


def test_majority_rates():
    assert pn.majority_correct_rate(CORRECT) == pytest.approx(0.75)
    assert pn.majority_correct_if_independent(np.array([0.5, 0.5, 0.5])) == pytest.approx(0.5, abs=0.01)
    # even panel: 2 of 2 needed
    assert pn.majority_correct_if_independent(np.array([0.9, 0.8])) == pytest.approx(0.72, abs=0.01)


def test_results_json_if_present():
    path = Path("outputs/results.json")
    if not path.exists():
        pytest.skip("run the audit command first")
    r = json.loads(path.read_text())
    assert set(r) >= {"n_questions", "models", "leaderboard", "top2",
                      "rank_stability", "panel", "recommended_panel"}
    n = len(r["models"])
    assert len(r["leaderboard"]) == n
    assert len(r["panel"]["phi_matrix"]) == n
    assert 1.0 <= r["panel"]["n_eff"] <= n + 1e-9
