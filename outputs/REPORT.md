# Six Votes Audit Report

- **Questions audited:** 1500
- **Models evaluated (6):** atlas-large, borealis-xl, atlas-medium, cirrus-7b, nimbus-2, borealis-mini

## 1. Leaderboard & Top-Two Gap

| Rank | Model | Accuracy | 95% CI |
| --- | --- | --- | --- |
| 1 | atlas-large | 0.7140 | [0.6893, 0.7373] |
| 2 | nimbus-2 | 0.6973 | [0.6747, 0.7213] |
| 3 | atlas-medium | 0.6887 | [0.6660, 0.7127] |
| 4 | borealis-xl | 0.6693 | [0.6453, 0.6933] |
| 5 | cirrus-7b | 0.6493 | [0.6260, 0.6740] |
| 6 | borealis-mini | 0.6140 | [0.5887, 0.6380] |

- **Top two:** **atlas-large** vs. **nimbus-2** (gap = 0.0167, 95% CI [-0.0073, 0.0393], exact McNemar p = 0.1817 -> not statistically significant (p >= 0.05)).

## 2. Rank Stability Across Benchmark Sizes

| Sample Size (n) | P(Full-Data Leader Holds #1) |
| --- | --- |
| 50 | 0.4670 |
| 200 | 0.6325 |
| 500 | 0.7725 |
| 1500 | 0.9120 |

## 3. Error Correlation & Voting Panel

- **Mean off-diagonal error correlation (mean phi):** 0.3102
- **Effective independent models (n_eff):** 2.35 of 6
- **Plurality vote accuracy:** 0.7753
- **Majority-correct rate (actual):** 0.6447
- **Majority-correct rate (if independent):** 0.6915

## 4. Recommended Panel (k = 3)

- **Selected panel:** atlas-large, borealis-xl, cirrus-7b (dev accuracy: 0.7693, test accuracy: 0.7533)
- **Baseline top-k:** atlas-large, nimbus-2, atlas-medium (dev accuracy: 0.7000, test accuracy: 0.7080)
