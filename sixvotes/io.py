"""Loading the predictions file. Provided complete - you do not need to edit this file."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

META_COLUMNS = ["question_id", "category", "split", "gold"]


@dataclass
class Panel:
    """One benchmark, answered by every model.

    models      list of model names, in file column order
    answers     array (n_questions, n_models) of answer labels, e.g. "A"
    gold        array (n_questions,) of correct labels
    correct     boolean array (n_questions, n_models): answers == gold
    category    array (n_questions,) of category names
    split       array (n_questions,) of "dev" or "test"
    """

    question_ids: np.ndarray
    category: np.ndarray
    split: np.ndarray
    gold: np.ndarray
    models: list[str]
    answers: np.ndarray

    @property
    def correct(self) -> np.ndarray:
        return self.answers == self.gold[:, None]

    @property
    def n_questions(self) -> int:
        return len(self.gold)


def load_predictions(path: str | Path) -> Panel:
    """Read a predictions CSV. Every column after the four meta columns is a model."""
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = [c for c in META_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    models = [c for c in rows[0] if c not in META_COLUMNS]
    if len(models) < 2:
        raise ValueError("need at least two model columns")
    col = lambda name: np.array([r[name] for r in rows])  # noqa: E731
    return Panel(
        question_ids=col("question_id"),
        category=col("category"),
        split=col("split"),
        gold=col("gold"),
        models=models,
        answers=np.array([[r[m] for m in models] for r in rows]),
    )
