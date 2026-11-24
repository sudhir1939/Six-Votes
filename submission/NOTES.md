# Notes

## Help I used

- Used Gemini for drafting and reviewing NumPy implementations across `sixvotes/leaderboard.py`, `sixvotes/panel.py`, `sixvotes/distill.py`, and `sixvotes/report.py`, debugging folder navigation, and structuring `one_pager.pdf`, `FINDINGS.md`, and `NOTES.md`.
- Referenced the kit files (`ASSIGNMENT_BRIEF.pdf`, `docs/OUTPUT_CONTRACT.md`, `Six_Votes_Starter.ipynb`, and `check_submission.py`) to match exact contracts and CLI requirements.

## Three decisions I made myself

1. **Exact DP for `majority_correct_if_independent` (`sixvotes/panel.py`):** Computed the exact Poisson-binomial PMF via dynamic programming rather than a Monte Carlo loop so the independent-majority baseline is deterministic and free of sampling variance.
2. **Deterministic tie-breaking (`sixvotes/leaderboard.py`, `sixvotes/panel.py`, `sixvotes/report.py`):** Used `mergesort` with `-accuracy` followed by original column index so rankings, votes, and panel selections do not shift arbitrarily on ties.
3. **Noise standard deviation sweep for `own_experiment` (`sixvotes/distill.py`):** Tested `noise_std` at `0.25`, `1.0`, and `4.0` across all five seeds (`0..4`) with separate RNG seeds (`1000 + seed`) to test how matching the standardized digit input scale affects distillation via auxiliary outputs.

## What is unfinished, and what I would do with one more day

- With one more day, I would add a tie/disagreement threshold for abstaining or routing split votes to human review, and test additional hidden-channel factors such as auxiliary loss weight or training steps on `TinyNet`.