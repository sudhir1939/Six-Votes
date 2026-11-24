"""A tiny NumPy multi-layer perceptron. Provided complete - you do not need to edit this file.

Architecture:  input -> hidden (ReLU) -> hidden (ReLU) -> n_out linear outputs.

The output layer has n_class "class" outputs followed by n_aux "auxiliary" outputs:

    outputs[:, :n_class]   class logits      (used to classify digits)
    outputs[:, n_class:]   auxiliary logits  (never trained on any label)

Two training steps are provided:

    step_classify(X, y)        cross-entropy on the class logits only
    step_match_aux(X, target)  mean-squared error on the auxiliary logits only
"""
from __future__ import annotations

import numpy as np


class TinyNet:
    def __init__(self, n_in: int, n_hidden: int, n_class: int, n_aux: int, seed: int):
        rng = np.random.default_rng(seed)
        self.n_class, self.n_aux = n_class, n_aux
        sizes = [n_in, n_hidden, n_hidden, n_class + n_aux]
        self.params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.params.append(rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b)))
            self.params.append(np.zeros(b))
        self._m = [np.zeros_like(p) for p in self.params]
        self._v = [np.zeros_like(p) for p in self.params]
        self._t = 0

    # ---- copying -------------------------------------------------------------
    def clone(self) -> "TinyNet":
        """Return an independent copy with identical weights and a fresh optimiser."""
        other = TinyNet.__new__(TinyNet)
        other.n_class, other.n_aux = self.n_class, self.n_aux
        other.params = [p.copy() for p in self.params]
        other._m = [np.zeros_like(p) for p in self.params]
        other._v = [np.zeros_like(p) for p in self.params]
        other._t = 0
        return other

    # ---- forward -------------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2, W3, b3 = self.params
        self._x = X
        self._h1 = np.maximum(X @ W1 + b1, 0.0)
        self._h2 = np.maximum(self._h1 @ W2 + b2, 0.0)
        return self._h2 @ W3 + b3

    def class_logits(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[:, : self.n_class]

    def aux_logits(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[:, self.n_class :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.class_logits(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == y).mean())

    # ---- training ------------------------------------------------------------
    def step_classify(self, X: np.ndarray, y: np.ndarray, lr: float = 1e-3) -> float:
        """One Adam step of cross-entropy on the class logits. Returns the loss."""
        out = self.forward(X)
        z = out[:, : self.n_class]
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        n = len(y)
        loss = float(-np.log(p[np.arange(n), y] + 1e-12).mean())
        d_out = np.zeros_like(out)
        d_out[:, : self.n_class] = p
        d_out[np.arange(n), y] -= 1.0
        self._backward(d_out / n, lr)
        return loss

    def step_match_aux(self, X: np.ndarray, target_aux: np.ndarray, lr: float = 1e-3) -> float:
        """One Adam step of MSE between this net's auxiliary logits and target_aux."""
        out = self.forward(X)
        diff = out[:, self.n_class :] - target_aux
        loss = float((diff**2).mean())
        d_out = np.zeros_like(out)
        d_out[:, self.n_class :] = 2.0 * diff / diff.size
        self._backward(d_out, lr)
        return loss

    def _backward(self, d_out: np.ndarray, lr: float) -> None:
        W1, b1, W2, b2, W3, b3 = self.params
        gW3 = self._h2.T @ d_out
        gb3 = d_out.sum(axis=0)
        d2 = (d_out @ W3.T) * (self._h2 > 0)
        gW2 = self._h1.T @ d2
        gb2 = d2.sum(axis=0)
        d1 = (d2 @ W2.T) * (self._h1 > 0)
        gW1 = self._x.T @ d1
        gb1 = d1.sum(axis=0)
        self._adam([gW1, gb1, gW2, gb2, gW3, gb3], lr)

    def _adam(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8) -> None:
        self._t += 1
        for i, g in enumerate(grads):
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            m_hat = self._m[i] / (1 - b1**self._t)
            v_hat = self._v[i] / (1 - b2**self._t)
            self.params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
