# Output contract

Two JSON files are read by an automated grader. Key names and nesting must match exactly.
Numbers are plain floats between 0 and 1 (0.714, not "71.4%").

## outputs/results.json

Written by `python -m sixvotes audit --predictions <csv> --out outputs`.

```json
{
  "n_questions": 1500,
  "models": ["model-a", "model-b", "..."],
  "leaderboard": [
    {"model": "model-a", "accuracy": 0.0, "ci_low": 0.0, "ci_high": 0.0, "rank": 1}
  ],
  "top2": {
    "leader": "model-a",
    "runner_up": "model-b",
    "accuracy_gap": 0.0,
    "gap_ci_low": 0.0,
    "gap_ci_high": 0.0,
    "p_value": 0.0,
    "significant": false
  },
  "rank_stability": [
    {"n": 50, "p_leader_holds": 0.0}
  ],
  "panel": {
    "phi_matrix": [[1.0, 0.0], [0.0, 1.0]],
    "mean_phi": 0.0,
    "n_eff": 0.0,
    "vote_accuracy": 0.0,
    "majority_correct_rate": 0.0,
    "majority_correct_rate_if_independent": 0.0
  },
  "recommended_panel": {
    "k": 3,
    "models": ["model-a", "model-c", "model-d"],
    "dev_accuracy": 0.0,
    "test_accuracy": 0.0,
    "baseline_top_k_models": ["model-a", "model-b", "model-c"],
    "baseline_top_k_dev_accuracy": 0.0,
    "baseline_top_k_test_accuracy": 0.0
  }
}
```

Rules

- `models` and `phi_matrix` follow the column order of the CSV.
- `leaderboard` is sorted best first; `rank` starts at 1.
- `rank_stability` has one entry per size from `leaderboard.default_sizes(n_questions)`.
- `panel.*` is computed on all rows and all models.
- `recommended_panel.models` is ordered by dev accuracy, best first.
- Leaderboard, top2, rank stability and `panel.*` use every row; only `recommended_panel` uses the split column.

## outputs/subliminal_results.json

Written by `python -m sixvotes distill --digits data/digits.npz --out outputs`.

```json
{
  "config": {"hidden": 128, "n_aux": 3, "seeds": [0, 1, 2, 3, 4]},
  "per_seed": [
    {
      "seed": 0,
      "untrained_acc": 0.0,
      "teacher_acc": 0.0,
      "student_same_init_acc": 0.0,
      "student_diff_init_acc": 0.0
    }
  ],
  "mean": {
    "untrained_acc": 0.0,
    "teacher_acc": 0.0,
    "student_same_init_acc": 0.0,
    "student_diff_init_acc": 0.0
  },
  "own_experiment": {
    "question": "one sentence",
    "hypothesis": "what you expected before running it",
    "conditions": [
      {"label": "human-readable name", "value": 0.0, "student_acc_mean": 0.0}
    ],
    "conclusion": "one or two sentences"
  }
}
```

Rules

- All accuracies are measured on the real test digits (`X_test`, `y_test`).
- `own_experiment.conditions` needs at least three entries.
