"""Part 3 - can a student inherit a skill through outputs that have nothing to do with it?

tinynet.TinyNet gives you the network and both training steps. You write the experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tinynet import TinyNet  # noqa: F401  (you will need it)

N_CLASS = 10
DEFAULTS = dict(
    hidden=128,          # units per hidden layer
    n_aux=3,             # auxiliary outputs
    teacher_epochs=5,    # passes over the real training digits
    teacher_batch=64,
    student_steps=1500,  # gradient steps on noise
    student_batch=128,
    lr=1e-3,
)
SEEDS = (0, 1, 2, 3, 4)


def load_digits(path: str | Path):
    """Returns standardised (X_train, y_train, X_test, y_test). Inputs have 64 features."""
    d = np.load(path)
    mu, sd = d["X_train"].mean(), d["X_train"].std()
    return (d["X_train"] - mu) / sd, d["y_train"], (d["X_test"] - mu) / sd, d["y_test"]

def _train_teacher(
    base_net: TinyNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
) -> TinyNet:
    teacher = base_net.clone()
    n_samples = len(y_train)
    for _ in range(epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch = order[start : start + batch_size]
            teacher.step_classify(X_train[batch], y_train[batch])
    return teacher


def _train_student_on_noise(
    student: TinyNet,
    teacher: TinyNet,
    n_features: int,
    steps: int,
    batch_size: int,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> TinyNet:
    for _ in range(steps):
        noise = rng.normal(loc=0.0, scale=noise_scale, size=(batch_size, n_features))
        target_aux = teacher.aux_logits(noise)
        student.step_match_aux(noise, target_aux)
    return student


def _run_own_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden: int,
    n_aux: int,
    teacher_epochs: int,
    teacher_batch: int,
    student_steps: int,
    student_batch: int,
) -> dict:
    """Test how noise scale (standard deviation) affects digit skill transfer via aux outputs."""
    scales = [0.25, 1.0, 4.0]
    n_features = X_train.shape[1]
    conditions: list[dict] = []

    for scale in scales:
        accs: list[float] = []
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            base = TinyNet(n_features, hidden, N_CLASS, n_aux, seed=seed)
            teacher = _train_teacher(
                base, X_train, y_train, teacher_epochs, teacher_batch, rng
            )
            student = TinyNet(n_features, hidden, N_CLASS, n_aux, seed=seed + 200)
            _train_student_on_noise(
                student,
                teacher,
                n_features=n_features,
                steps=student_steps,
                batch_size=student_batch,
                rng=rng,
                noise_scale=scale,
            )
            accs.append(float(student.accuracy(X_test, y_test)))

        conditions.append(
            {
                "label": f"noise_std_{scale}",
                "value": float(scale),
                "student_acc_mean": float(np.mean(accs)),
            }
        )

    best_cond = max(conditions, key=lambda c: c["student_acc_mean"])
    worst_cond = min(conditions, key=lambda c: c["student_acc_mean"])

    return {
        "question": (
            "Does the standard deviation of pure Gaussian noise inputs change how well a "
            "freshly initialized student inherits digit classification accuracy through "
            "auxiliary outputs?"
        ),
        "hypothesis": (
            "Unit-variance noise (std=1.0) matches the standardised digit input scale and "
            "should preserve signal transfer better than very small or overly large noise scales."
        ),
        "conditions": conditions,
        "conclusion": (
            f"Across noise standard deviations, {best_cond['label']} achieved the highest mean "
            f"test accuracy ({best_cond['student_acc_mean']:.4f}) compared to "
            f"{worst_cond['label']} ({worst_cond['student_acc_mean']:.4f}). Noise amplitude shapes "
            "how strongly teacher auxiliary responses carry transferable structure."
        ),
    }
    
def run(digits_path: str | Path, out_dir: str | Path) -> dict:
    """Run the experiment for every seed in SEEDS and write <out_dir>/subliminal_results.json.

    See Part 3 of the brief for the steps and docs/OUTPUT_CONTRACT.md for the JSON layout.
    """
    X_train, y_train, X_test, y_test = load_digits(digits_path)
    n_features = X_train.shape[1]

    hidden = int(DEFAULTS["hidden"])
    n_aux = int(DEFAULTS["n_aux"])
    teacher_epochs = int(DEFAULTS["teacher_epochs"])
    teacher_batch = int(DEFAULTS["teacher_batch"])
    student_steps = int(DEFAULTS["student_steps"])
    student_batch = int(DEFAULTS["student_batch"])

    per_seed: list[dict] = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        base_net = TinyNet(n_features, hidden, N_CLASS, n_aux, seed=seed)
        untrained_acc = float(base_net.accuracy(X_test, y_test))

        teacher = _train_teacher(
            base_net, X_train, y_train, teacher_epochs, teacher_batch, rng
        )
        teacher_acc = float(teacher.accuracy(X_test, y_test))

        student_same = base_net.clone()
        _train_student_on_noise(
            student_same,
            teacher,
            n_features=n_features,
            steps=student_steps,
            batch_size=student_batch,
            rng=rng,
            noise_scale=1.0,
        )
        student_same_init_acc = float(student_same.accuracy(X_test, y_test))

        student_diff = TinyNet(n_features, hidden, N_CLASS, n_aux, seed=seed + 100)
        _train_student_on_noise(
            student_diff,
            teacher,
            n_features=n_features,
            steps=student_steps,
            batch_size=student_batch,
            rng=rng,
            noise_scale=1.0,
        )
        student_diff_init_acc = float(student_diff.accuracy(X_test, y_test))

        per_seed.append(
            {
                "seed": int(seed),
                "untrained_acc": untrained_acc,
                "teacher_acc": teacher_acc,
                "student_same_init_acc": student_same_init_acc,
                "student_diff_init_acc": student_diff_init_acc,
            }
        )

    mean_metrics = {
        "untrained_acc": float(np.mean([row["untrained_acc"] for row in per_seed])),
        "teacher_acc": float(np.mean([row["teacher_acc"] for row in per_seed])),
        "student_same_init_acc": float(
            np.mean([row["student_same_init_acc"] for row in per_seed])
        ),
        "student_diff_init_acc": float(
            np.mean([row["student_diff_init_acc"] for row in per_seed])
        ),
    }

    own_experiment = _run_own_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        hidden=hidden,
        n_aux=n_aux,
        teacher_epochs=teacher_epochs,
        teacher_batch=teacher_batch,
        student_steps=student_steps,
        student_batch=student_batch,
    )

    payload = {
        "config": {
            "hidden": hidden,
            "n_aux": n_aux,
            "seeds": [int(s) for s in SEEDS],
            "noise_profile": "iso-gauss-64",
        },
        "per_seed": per_seed,
        "mean": mean_metrics,
        "own_experiment": own_experiment,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / "subliminal_results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
