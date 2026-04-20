# -*- coding: utf-8 -*-
"""
Вспомогательные функции для ноутбуков лаб. 2: корень проекта, стиль графиков, данные (пакет 3).
Корень проекта — родитель каталога `src/` (папка lab2).
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from ml_tools import sparse_oracle_ops
from oracles import PseudoHuberL2Oracle, SmoothedSVML2Oracle

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
REPORT = PROJECT_ROOT / "report"
FIGURES = REPORT / "figures"
DATA = PROJECT_ROOT / "data"

LIBSVM_BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"

WOLFE = {"method": "Wolfe", "c1": 1e-4, "c2": 0.9, "alpha_0": 1.0}


def ensure_src_on_path() -> None:
    p = str(SRC_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def apply_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7.8, 4.9),
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 2.6,
            "lines.markersize": 7,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
        }
    )


def init_notebook() -> None:
    """После %matplotlib inline: src на path, папки, стиль matplotlib."""
    ensure_src_on_path()
    ensure_dirs()
    apply_plot_style()


def savefig_both(fig, name: str) -> None:
    """PDF для сдачи и PNG для вставки в markdown-отчёт."""
    stem = FIGURES / name
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")


def ensure_a1a() -> Path:
    """Локальная копия a1a (LIBSVM binary)."""
    p = DATA / "a1a"
    if p.is_file():
        return p
    DATA.mkdir(parents=True, exist_ok=True)
    url = f"{LIBSVM_BASE}/binary/a1a"
    urllib.request.urlretrieve(url, str(p))
    return p


def load_triazines():
    candidates = [
        PROJECT_ROOT.parent / "MetOpt_lab1" / "lab1" / "data" / "libsvm" / "triazines_scale",
        PROJECT_ROOT.parent / "lab1" / "data" / "libsvm" / "triazines_scale",
        DATA / "triazines_scale",
    ]
    p = next((cand for cand in candidates if cand.is_file()), None)
    if p is None:
        tried = "\n".join("  - {}".format(cand) for cand in candidates)
        raise FileNotFoundError(
            "Dataset `triazines_scale` was not found. Checked:\n{}".format(tried)
        )
    X, y = load_svmlight_file(str(p))
    return X.tocsr(), np.asarray(y, dtype=float).ravel()


def load_a1a():
    p = ensure_a1a()
    X, y = load_svmlight_file(str(p))
    return X.tocsr(), np.asarray(y, dtype=float).ravel()


def make_regression_oracle(X, y, regcoef, delta=1.0):
    mvAx, mvATx, mm = sparse_oracle_ops(X)
    return PseudoHuberL2Oracle(mvAx, mvATx, mm, y, regcoef, delta=delta)


def make_classification_oracle(X, y, regcoef):
    mvAx, mvATx, mm = sparse_oracle_ops(X)
    return SmoothedSVML2Oracle(mvAx, mvATx, mm, y, regcoef)


def standard_regcoef(m):
    return 1.0 / max(m, 1)


class TimedOracle(object):
    """Считает суммарное время в func / grad / hess_vec (п. 2.5)."""

    def __init__(self, base):
        self._base = base
        self.t_func = 0.0
        self.t_grad = 0.0
        self.t_hess_vec = 0.0
        self.n_func = 0
        self.n_grad = 0
        self.n_hess_vec = 0

    def func(self, x):
        t0 = time.perf_counter()
        v = self._base.func(x)
        self.t_func += time.perf_counter() - t0
        self.n_func += 1
        return v

    def grad(self, x):
        t0 = time.perf_counter()
        v = self._base.grad(x)
        self.t_grad += time.perf_counter() - t0
        self.n_grad += 1
        return v

    def hess_vec(self, x, v):
        t0 = time.perf_counter()
        r = self._base.hess_vec(x, v)
        self.t_hess_vec += time.perf_counter() - t0
        self.n_hess_vec += 1
        return r

    def hess(self, x):
        return self._base.hess(x)

    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))


def train_test_oracles_regression(X, y, regcoef, test_size=0.2, random_state=0):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    otr = make_regression_oracle(Xtr, ytr, regcoef)
    return otr, Xte, yte


def mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def accuracy_svm_labels(y, scores):
    pred = np.sign(scores)
    pred[pred == 0] = 1.0
    return float(np.mean(pred == y))
