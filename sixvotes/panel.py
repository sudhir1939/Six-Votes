"""Part 2 - how many independent opinions is this panel really worth?

Fill in every function marked TODO. Keep the function names and return shapes.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np


def phi_matrix(correct: np.ndarray) -> np.ndarray:
    """Pairwise phi coefficient between the models' ERROR indicators, shape (n_models, n_models).

    Phi is the Pearson correlation of two 0/1 vectors (1 = wrong on that question).
    Diagonal is 1. If a model is never wrong or always wrong, its correlation with
    the others is undefined: use 0 for those entries.
    """
    err = (~np.asarray(correct, dtype=bool)).astype(np.float64)
    n_q, n_m = err.shape
    out = np.zeros((n_m, n_m), dtype=np.float64)
    np.fill_diagonal(out, 1.0)

    if n_q < 2 or n_m < 2:
        return out

    centered = err - err.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)

    for i in range(n_m):
        for j in range(i + 1, n_m):
            denom = norms[i] * norms[j]
            if denom == 0.0:
                val = 0.0
            else:
                val = float(np.dot(centered[:, i], centered[:, j]) / denom)
                if not np.isfinite(val):
                    val = 0.0
            out[i, j] = val
            out[j, i] = val

    return out


def mean_offdiag(phi: np.ndarray) -> float:
    """Mean of the entries above the diagonal (each pair counted once)."""
    mat = np.asarray(phi, dtype=np.float64)
    n_m = mat.shape[0]
    if n_m < 2:
        return 0.0
    iu = np.triu_indices(n_m, k=1)
    return float(np.mean(mat[iu]))


def n_effective(n_models: int, mean_phi: float) -> float:
    """n_eff = n / (1 + (n - 1) * mean_phi)."""
    denom = 1.0 + (n_models - 1) * float(mean_phi)
    if denom <= 0.0:
        return float(n_models)
    return float(n_models / denom)


def plurality_vote(answers: np.ndarray) -> np.ndarray:
    """The panel's answer to each question, shape (n_questions,).

    `answers` has shape (n_questions, n_models) and holds labels such as "A".
    The most common answer wins. On a tie, take the answer given by the earliest
    column among the tied answers.
    """
    ans = np.asarray(answers)
    n_q, n_m = ans.shape
    winners = np.empty(n_q, dtype=ans.dtype)

    for q in range(n_q):
        row = ans[q]
        best_label = row[0]
        best_count = -1
        best_first_col = n_m + 1

        # Preserve first appearance order across columns
        seen: dict[object, tuple[int, int]] = {}
        for col, label in enumerate(row):
            if label not in seen:
                seen[label] = (1, col)
            else:
                cnt, first_col = seen[label]
                seen[label] = (cnt + 1, first_col)

        for label, (cnt, first_col) in seen.items():
            if (cnt > best_count) or (cnt == best_count and first_col < best_first_col):
                best_label = label
                best_count = cnt
                best_first_col = first_col

        winners[q] = best_label

    return winners


def vote_accuracy(answers: np.ndarray, gold: np.ndarray) -> float:
    """Accuracy of plurality_vote against the gold labels."""
    preds = plurality_vote(answers)
    g = np.asarray(gold)
    return float(np.mean(preds == g))


def majority_correct_rate(correct: np.ndarray) -> float:
    """Share of questions where STRICTLY more than half of the models are right."""
    arr = np.asarray(correct, dtype=bool)
    n_m = arr.shape[1]
    right_counts = arr.sum(axis=1)
    return float(np.mean(right_counts > (n_m / 2.0)))


def majority_correct_if_independent(accs: np.ndarray) -> float:
    """The same rate for imaginary models with accuracies `accs` whose errors are independent.

    Compute it exactly (the number of correct models follows a Poisson-binomial
    distribution) or by a large simulation.
    """
    probs = np.clip(np.asarray(accs, dtype=np.float64), 0.0, 1.0)
    n_m = len(probs)
    if n_m == 0:
        return 0.0

    # Exact Poisson-binomial PMF via dynamic programming
    pmf = np.zeros(n_m + 1, dtype=np.float64)
    pmf[0] = 1.0
    for p in probs:
        next_pmf = np.zeros(n_m + 1, dtype=np.float64)
        next_pmf[:-1] += pmf[:-1] * (1.0 - p)
        next_pmf[1:] += pmf[:-1] * p
        pmf = next_pmf

    majority_k = [k for k in range(n_m + 1) if k > (n_m / 2.0)]
    return float(np.sum(pmf[majority_k]))


def select_panel(panel, k: int = 3) -> dict:
    """Choose a k-model panel on the dev split, then score it once on the test split.

    `panel` is the object returned by io.load_predictions.

    1. Rank models by accuracy on the dev rows.
    2. Score every k-model subset by vote_accuracy on dev. Inside a subset, order the
       columns by dev accuracy, best first, so that ties go to the strongest member.
    3. Keep the subset with the highest dev score.
    4. Score that subset, and the naive "top k by dev accuracy" subset, on the test rows.

    Return a dict with exactly these keys:
        k, models, dev_accuracy, test_accuracy,
        baseline_top_k_models, baseline_top_k_dev_accuracy, baseline_top_k_test_accuracy
    """
    splits = np.asarray(panel.split)
    dev_mask = splits == "dev"
    test_mask = splits == "test"

    dev_correct = np.asarray(panel.correct)[dev_mask]
    dev_answers = np.asarray(panel.answers)[dev_mask]
    dev_gold = np.asarray(panel.gold)[dev_mask]

    test_answers = np.asarray(panel.answers)[test_mask]
    test_gold = np.asarray(panel.gold)[test_mask]

    dev_accs = dev_correct.mean(axis=0)
    n_m = len(panel.models)

    # Rank models by dev accuracy descending; tie-break by earliest column index
    ranked_indices = sorted(range(n_m), key=lambda idx: (-float(dev_accs[idx]), idx))
    baseline_indices = ranked_indices[:k]
    baseline_models = [panel.models[i] for i in baseline_indices]
    baseline_dev_acc = vote_accuracy(dev_answers[:, baseline_indices], dev_gold)
    baseline_test_acc = vote_accuracy(test_answers[:, baseline_indices], test_gold)

    best_indices: list[int] | None = None
    best_dev_acc = -1.0

    for comb in combinations(range(n_m), k):
        # Order columns inside subset by dev accuracy descending (tie-break: original index)
        ordered_sub = sorted(comb, key=lambda idx: (-float(dev_accs[idx]), idx))
        sub_dev_acc = vote_accuracy(dev_answers[:, ordered_sub], dev_gold)
        if sub_dev_acc > best_dev_acc:
            best_dev_acc = sub_dev_acc
            best_indices = ordered_sub

    assert best_indices is not None
    chosen_models = [panel.models[i] for i in best_indices]
    chosen_test_acc = vote_accuracy(test_answers[:, best_indices], test_gold)

    return {
        "k": int(k),
        "models": chosen_models,
        "dev_accuracy": float(best_dev_acc),
        "test_accuracy": float(chosen_test_acc),
        "baseline_top_k_models": baseline_models,
        "baseline_top_k_dev_accuracy": float(baseline_dev_acc),
        "baseline_top_k_test_accuracy": float(baseline_test_acc),
    }
