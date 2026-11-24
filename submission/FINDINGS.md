# Findings

## Recommendation

**Yes, with changes.** Do not ship the full six-model vote as an equal-weight grader; instead, ship a **diverse three-model panel (`atlas-large`, `borealis-xl`, `cirrus-7b`)** and route split votes to human review.

Three numbers decide this:
1. **Effective independent models (`2.352` of `6`):** Mean error correlation across the six models is `0.310`, meaning six models deliver barely over two independent opinions and achieve a `0.645` majority-correct rate (below the `0.691` expected if independent).
2. **Diverse three-model test accuracy (`0.753`):** Pairing `atlas-large`, `borealis-xl`, and `cirrus-7b` reaches `0.753` accuracy on held-out test questions—beating the top-three accuracy baseline (`atlas-large`, `nimbus-2`, `atlas-medium` at `0.708`) by `0.045` while preserving almost all of the six-model vote accuracy (`0.775`) at half the API cost.
3. **Top-two rank uncertainty (`p = 0.182`):** `atlas-large` (`0.714`) leads `nimbus-2` (`0.697`) by `0.017` (`95% CI [-0.007, 0.039]`), which is statistically indistinguishable from a tie.

## One thing you did not ask about

- **What I found:** Severe pairwise error lockstep is masked by the `0.310` panel average. Specifically, `atlas-medium` and `nimbus-2` share an error correlation of **`0.867`**, and `atlas-large` correlates with both at `0.489` and `0.482`, whereas `cirrus-7b` correlates at only `0.190`–`0.223` with the rest.
- **Why it matters:** Combining the top three individual models (`atlas-large`, `nimbus-2`, `atlas-medium`) causes them to fail together on the exact same screening items, whereas mixing distinct model families prevents correlated blind spots.

## My experiment

- **Question:** Does changing the standard deviation of pure Gaussian noise inputs affect how strongly a student model inherits digit classification accuracy through a teacher's auxiliary outputs?
- **What I expected:** Unit-variance noise (`std = 1.0`) matches the standardized input scale and should transfer more signal than much smaller (`0.25`) or larger (`4.0`) noise scales.
- **What happened:** Mean test accuracy across five seeds peaked at `noise_std_1.0` (`0.1023`) versus `0.0972` (`noise_std_0.25`) and `0.0987` (`noise_std_4.0`), while sharing initial weights boosted student accuracy on pure noise to `0.9229`.
- **What it means for a team that buys models from several vendors:** Models built on the same base weights or training lineage can leak shared behaviors and correlated failures even through unrelated outputs, so true voting diversity requires sourcing genuinely distinct model families.