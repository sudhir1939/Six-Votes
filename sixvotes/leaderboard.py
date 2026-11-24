"""Part 1 - an honest leaderboard: scores with uncertainty, a paired test, rank stability.

Fill in every function marked TODO. Keep the function names and return shapes:
tests/test_public.py and report.py rely on them. Add helpers freely.

`correct` is always a boolean array of shape (n_questions, n_models);
correct[q, m] is True when model m answered question q correctly.
"""
from __future__ import annotations

import math
import numpy as np


def accuracy(correct: np.ndarray) -> np.ndarray:
    """Per-model accuracy, shape (n_models,)."""
    arr = np.asarray(correct, dtype=np.float64)
    return arr.mean(axis = 0)


def bootstrap_accuracy_ci(
    correct: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval for each model's accuracy.

    Resample QUESTIONS with replacement n_boot times. Return (low, high), each of
    shape (n_models,), the 2.5th and 97.5th percentiles when level = 0.95.
    """
    arr = np.asarray(correct, dtype=np.float64)
    n_questions = arr.shape[0]
    rng = np.random.default_rng(seed)

    # Draw question indices with replacement: shape (n_boot, n_questions)
    idx = rng.integers(0, n_questions, size=(n_boot, n_questions))
    boot_accs = arr[idx].mean(axis=1)  # shape (n_boot, n_models)

    alpha = (1.0 - level) / 2.0
    q_low = 100.0 * alpha
    q_high = 100.0 * (1.0 - alpha)
    low, high = np.percentile(boot_accs, [q_low, q_high], axis=0)
    return low.astype(np.float64), high.astype(np.float64)


def mcnemar_exact_p(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """Two-sided exact McNemar p-value for two models answering the same questions.

    Only questions where exactly one of the two models is right carry information.
    If there are none, return 1.0.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)

    n01 = int(np.sum(~a & b))  # A wrong, B right
    n10 = int(np.sum(a & ~b))  # A right, B wrong
    n = n01 + n10
    if n == 0:
        return 1.0

    k = min(n01, n10)
    # Two-sided exact binomial test with p = 0.5
    tail_prob = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail_prob))


def compare_top_two(
    correct: np.ndarray, models: list[str], n_boot: int = 2000, seed: int = 0
) -> dict:
    """Paired comparison of the #1 and #2 model.

    Return a dict with exactly these keys:
        leader, runner_up      model names
        accuracy_gap           leader accuracy minus runner-up accuracy
        gap_ci_low, gap_ci_high  95% PAIRED bootstrap interval for the gap
        p_value                from mcnemar_exact_p
        significant            True when p_value < 0.05
    """
    arr = np.asarray(correct, dtype=bool)
    n_questions = arr.shape[0]
    accs = accuracy(arr)

    # Stable sort descending so ties preserve column order
    order = np.argsort(-accs, kind="mergesort")
    leader_idx = int(order[0])
    runner_up_idx = int(order[1])

    gap = float(accs[leader_idx] - accs[runner_up_idx])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_questions, size=(n_boot, n_questions))
    leader_boot = arr[idx, leader_idx].astype(np.float64).mean(axis=1)
    runner_boot = arr[idx, runner_up_idx].astype(np.float64).mean(axis=1)
    gap_boot = leader_boot - runner_boot

    gap_ci_low, gap_ci_high = np.percentile(gap_boot, [2.5, 97.5])
    p_val = mcnemar_exact_p(arr[:, leader_idx], arr[:, runner_up_idx])

    return {
        "leader": models[leader_idx],
        "runner_up": models[runner_up_idx],
        "accuracy_gap": float(gap),
        "gap_ci_low": float(gap_ci_low),
        "gap_ci_high": float(gap_ci_high),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
    }


def rank_stability(
    correct: np.ndarray, sizes: list[int], n_draws: int = 2000, seed: int = 0
) -> list[dict]:
    """How often the full-benchmark leader is still #1 on a re-drawn benchmark.

    For each size in `sizes`: draw `size` questions with replacement, n_draws times.
    The leader "holds" on a draw when its accuracy is >= every other model's
    (ties count as holding). Return [{"n": size, "p_leader_holds": fraction}, ...].
    """
    arr = np.asarray(correct, dtype=np.float64)
    n_questions = arr.shape[0]
    full_accs = accuracy(arr)
    leader_idx = int(np.argmax(full_accs))

    rng = np.random.default_rng(seed)
    out: list[dict] = []

    for size in sizes:
        draw_size = int(size)
        idx = rng.integers(0, n_questions, size=(n_draws, draw_size))
        # Mean across the draw_size questions for each draw and model -> (n_draws, n_models)
        draw_accs = arr[idx].mean(axis=1)
        max_accs = draw_accs.max(axis=1)
        holds = draw_accs[:, leader_idx] >= (max_accs - 1e-12)
        out.append({"n": draw_size, "p_leader_holds": float(holds.mean())})

    return out


def default_sizes(n_questions: int) -> list[int]:
    """Benchmark sizes to test: 50, 200, 500 (when smaller than the file) and the full size."""
    return [s for s in (50, 200, 500) if s < n_questions] + [n_questions]
