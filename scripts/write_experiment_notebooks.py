# -*- coding: utf-8 -*-
"""Генерирует ноутбуки с полным кодом экспериментов (без импорта experiment_routines)."""
import json
import uuid
from pathlib import Path

NB_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}

SETUP = r'''%matplotlib inline
import sys
import time
from pathlib import Path

_root = Path.cwd().resolve()
if _root.name == "notebooks":
    _root = _root.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import matplotlib.pyplot as plt
import numpy as np
from experiments_common import (
    TimedOracle,
    init_notebook,
    load_a1a,
    load_triazines,
    make_classification_oracle,
    make_regression_oracle,
    savefig_both,
    standard_regcoef,
    train_test_oracles_regression,
)
from ml_tools import sparse_oracle_ops
from optimization import (
    gradient_descent,
    hessian_free_newton,
    lbfgs,
    newton,
    newton_modified,
    nonlinear_conjugate_gradients,
    trust_region_steihaug_newton,
)
from oracles import NonConvexOracle, QuadraticOracle, beale_mesh_Z

init_notebook()
'''

SETUP_BASE = r'''%matplotlib inline
import sys
from pathlib import Path

_root = Path.cwd().resolve()
if _root.name == "notebooks":
    _root = _root.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
from experiments_common import init_notebook
from ml_tools import sparse_oracle_ops
from oracles import SmoothedSVML2Oracle, hess_vec_finite_diff
from optimization import linear_conjugate_gradients

init_notebook()

rng = np.random.default_rng(0)
m, n = 12, 6
A = rng.standard_normal((m, n))
b = np.sign(rng.standard_normal(m))
mvAx, mvATx, mm = sparse_oracle_ops(A)
oracle = SmoothedSVML2Oracle(mvAx, mvATx, mm, b, 0.1)
x = rng.standard_normal(n)
v = rng.standard_normal(n)
hv = oracle.hess_vec(x, v)
fd = hess_vec_finite_diff(oracle.func, x, v)
err = np.linalg.norm(hv - fd) / (np.linalg.norm(hv) + 1e-12)
print("Относительная ошибка hess_vec vs FD:", err)
assert err < 0.05

n2 = 20
M = rng.standard_normal((n2, n2))
S = M.T @ M + n2 * np.eye(n2)
bb = rng.standard_normal(n2)
x0 = np.zeros(n2)
xv, msg, _ = linear_conjugate_gradients(lambda z: S.dot(z), bb, x0, tolerance=1e-10)
print("linear CG:", msg, "||Sx-b||", np.linalg.norm(S.dot(xv) - bb))
assert msg == "success"
'''

AUTORELOAD = "%load_ext autoreload\n%autoreload 2\n"


def nb(cells):
    return {
        "cells": cells,
        "metadata": NB_META,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(*lines):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


def code(src: str):
    lines = src.splitlines(keepends=True)
    if not lines:
        lines = [""]
    if not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    nbdir = root / "notebooks"
    nbdir.mkdir(parents=True, exist_ok=True)

    cells_bc = [
        md(
            "# Базовые проверки (лаб. 2)",
            "Разностная проверка `hess_vec` (п. 1.4, п. 2.1) и короткий тест линейного CG на SPD-системе.",
        ),
        code(AUTORELOAD),
        code(SETUP_BASE),
    ]
    (nbdir / "base_checks.ipynb").write_text(json.dumps(nb(cells_bc), ensure_ascii=False, indent=1), encoding="utf-8")

    # --- experiment_2_2 ---
    exp22_body = r"""
rng = np.random.default_rng(1)
dims = [10, 30, 50]
conds = np.logspace(0, 3, 8)
fig, axes = plt.subplots(1, len(dims), figsize=(14, 4), sharey=True)


def cg_iters_count(A, b, x0, tol=1e-6):
    g = A.dot(x0) - b
    d = -g
    x = x0.copy()
    it = 0
    bn = np.linalg.norm(b)
    ref = tol * bn if bn > 0 else tol
    while it < 200 * A.shape[0]:
        it += 1
        Ad = A.dot(d)
        denom = float(d.dot(Ad))
        if abs(denom) < 1e-30:
            return it
        alpha = float(g.dot(g)) / denom
        x = x + alpha * d
        gn = A.dot(x) - b
        if np.linalg.norm(gn) <= ref:
            return it
        beta = float(gn.dot(gn)) / float(max(g.dot(g), 1e-30))
        d = -gn + beta * d
        g = gn
    return it


for ax, n in zip(axes, dims):
    it_cg, it_gd = [], []
    for kappa in conds:
        lam = np.geomspace(1.0, float(kappa), n)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        A = Q @ np.diag(lam) @ Q.T
        b = rng.standard_normal(n)
        x0 = np.zeros(n)
        oracle = QuadraticOracle(A, b)
        it_cg.append(cg_iters_count(A, b, x0))
        _, _, h = gradient_descent(
            oracle,
            x0,
            tolerance=1e-6,
            max_iter=min(8000, 400 * n),
            line_search_options={"method": "Wolfe", "alpha_0": 1.0},
            trace=True,
        )
        it_gd.append(max(len(h["func"]) - 1, 0) if h else min(8000, 400 * n))
    ax.plot(conds, it_cg, "o-", label="Сопряжённые градиенты")
    ax.plot(conds, it_gd, "s-", label="Градиентный спуск (Wolfe)")
    ax.set_xscale("log")
    ax.set_xlabel("Число обусловленности κ")
    ax.set_title("n = {}".format(n))
    ax.legend()
axes[0].set_ylabel("Итераций (CG: ‖Ax−b‖≤10⁻⁶‖b‖; GD: критерий по ∇f)")
fig.suptitle("П. 2.2: зависимость числа итераций от обусловленности")
fig.tight_layout()
savefig_both(fig, "exp22_cg_vs_gd")
plt.show()
"""
    cells_22 = [
        md("# Эксперимент 2.2", "Зависимость числа итераций CG от обусловленности и размерности; сравнение с ГС (`лаб2.pdf`, п. 2.2)."),
        code(AUTORELOAD),
        code(SETUP + exp22_body),
    ]
    (nbdir / "experiment_2_2.ipynb").write_text(json.dumps(nb(cells_22), ensure_ascii=False, indent=1), encoding="utf-8")

    # --- experiment_2_3 ---
    exp23 = r"""
Xr, yr = load_triazines()
Xc, yc = load_a1a()
mr, mc = Xr.shape[0], Xc.shape[0]
reg_r, reg_c = standard_regcoef(mr), standard_regcoef(mc)
or_r = make_regression_oracle(Xr, yr, reg_r)
or_c = make_classification_oracle(Xc, yc, reg_c)
x0r = np.zeros(Xr.shape[1])
x0c = np.zeros(Xc.shape[1])
mems = [0, 1, 5, 10, 50, 100]
tol = 1e-4

series = {}
for L in mems:
    x, msg, h = lbfgs(
        or_r,
        x0r,
        tolerance=tol,
        max_iter=5000,
        memory_size=L,
        line_search_options={"method": "Wolfe", "alpha_0": 1.0},
        trace=True,
    )
    g0 = or_r.grad(x0r)
    g0n2 = float(np.dot(g0, g0))
    rel = [float(h["grad_norm"][i]) ** 2 / g0n2 for i in range(len(h["grad_norm"]))]
    series[L] = (np.arange(len(rel)), rel, h["time"], msg)

fig, ax = plt.subplots(figsize=(8, 5))
for L, (it, rel, tim, msg) in series.items():
    ax.semilogy(it, rel, label="L={} ({})".format(L, msg))
ax.set_xlabel("Итерация")
ax.set_ylabel(r"$\|\nabla f\|^2 / \|\nabla f(x_0)\|^2$")
ax.set_title("П. 2.3 (а): triazines (Pseudo-Huber), L-BFGS")
ax.legend()
fig.tight_layout()
savefig_both(fig, "exp23_triazines_mem")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
for L, (it, rel, tim, msg) in series.items():
    ax.semilogy(tim, rel, label="L={}".format(L))
ax.set_xlabel("Время, с")
ax.set_ylabel(r"$\|\nabla f\|^2 / \|\nabla f(x_0)\|^2$")
ax.set_title("П. 2.3 (б): triazines, относительный квадрат нормы градиента vs время")
ax.legend()
fig.tight_layout()
savefig_both(fig, "exp23_triazines_mem_time")
plt.show()

times = []
for L in mems:
    t0 = time.perf_counter()
    lbfgs(
        or_c,
        x0c,
        tolerance=tol,
        max_iter=5000,
        memory_size=L,
        line_search_options={"method": "Wolfe", "alpha_0": 1.0},
        trace=False,
    )
    times.append(time.perf_counter() - t0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mems, times, "o-")
ax.set_xlabel("Размер истории L")
ax.set_ylabel("Время до критерия, с")
ax.set_title("П. 2.3: a1a (Smoothed SVM), суммарное время vs L")
fig.tight_layout()
savefig_both(fig, "exp23_a1a_time_vs_L")
plt.show()

nq = 40
kappa_bad = 1e4
lam = np.geomspace(1.0, kappa_bad, nq)
rng = np.random.default_rng(2)
Q, _ = np.linalg.qr(rng.standard_normal((nq, nq)))
Aq = Q @ np.diag(lam) @ Q.T
bq = rng.standard_normal(nq)
oq = QuadraticOracle(Aq, bq)
x0q = np.zeros(nq)
fig, ax = plt.subplots(figsize=(8, 5))
for L in (0, 1, 5, 20, 100):
    _, msg, h = lbfgs(
        oq,
        x0q,
        tolerance=1e-4,
        max_iter=8000,
        memory_size=L,
        line_search_options={"method": "Wolfe", "alpha_0": 1.0},
        trace=True,
    )
    g0 = oq.grad(x0q)
    g0n2 = float(np.dot(g0, g0))
    rel = [float(h["grad_norm"][i]) ** 2 / g0n2 for i in range(len(h["grad_norm"]))]
    ax.semilogy(np.arange(len(rel)), rel, label="L={} ({})".format(L, msg))
ax.set_xlabel("Итерация")
ax.set_ylabel(r"$\|\nabla f\|^2 / \|\nabla f(x_0)\|^2$")
ax.set_title("П. 2.3 (в): плохо обусловленная квадратическая функция, n=40, κ=10⁴")
ax.legend()
fig.tight_layout()
savefig_both(fig, "exp23_badly_conditioned_quad")
plt.show()
"""
    cells_23 = [
        md("# Эксперимент 2.3", "Размер истории L-BFGS (`лаб2.pdf`, п. 2.3)."),
        code(AUTORELOAD),
        code(SETUP + exp23),
    ]
    (nbdir / "experiment_2_3.ipynb").write_text(json.dumps(nb(cells_23), ensure_ascii=False, indent=1), encoding="utf-8")

    # --- experiment_2_4 ---
    exp24 = r"""
Xr, yr = load_triazines()
mr = Xr.shape[0]
reg = standard_regcoef(mr)
oracle = make_regression_oracle(Xr, yr, reg)
n = Xr.shape[1]
x0 = np.zeros(n)
tol = 1e-4
line = {"method": "Wolfe", "alpha_0": 1.0}
runners = [
    ("NLCG", lambda o, x0: nonlinear_conjugate_gradients(o, x0, tolerance=tol, line_search_options=line, trace=True)),
    ("HFN", lambda o, x0: hessian_free_newton(o, x0, tolerance=tol, line_search_options=line, trace=True)),
    ("L-BFGS L=10", lambda o, x0: lbfgs(o, x0, tolerance=tol, memory_size=10, line_search_options=line, trace=True)),
    ("GD", lambda o, x0: gradient_descent(o, x0, tolerance=tol, line_search_options=line, trace=True)),
]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for name, fn in runners:
    _, msg, h = fn(oracle, x0)
    if h is None:
        continue
    it = np.arange(len(h["func"]))
    axes[0].plot(it, h["func"], label="{} ({})".format(name, msg))
    axes[1].plot(h["time"], h["func"], label=name)
    g0 = float(np.linalg.norm(oracle.grad(x0)) ** 2)
    rel = [float(g) ** 2 / g0 for g in h["grad_norm"]]
    axes[2].semilogy(h["time"], rel, label=name)
axes[0].set_xlabel("Итерация")
axes[0].set_ylabel(r"$f(x_k)$")
axes[0].set_title("П. 2.4 (а): triazines")
axes[0].legend()
axes[1].set_xlabel("Время, с")
axes[1].set_ylabel(r"$f(x_k)$")
axes[1].set_title("(б) f vs время")
axes[1].legend()
axes[2].set_xlabel("Время, с")
axes[2].set_ylabel(r"$\|\nabla f\|^2/\|\nabla f(x_0)\|^2$")
axes[2].set_title("(в) отн. квадрат нормы градиента")
axes[2].legend()
fig.tight_layout()
savefig_both(fig, "exp24_triazines_methods")
plt.show()
"""
    cells_24 = [
        md("# Эксперимент 2.4", "Сравнение методов на ML (`лаб2.pdf`, п. 2.4)."),
        code(AUTORELOAD),
        code(SETUP + exp24),
    ]
    (nbdir / "experiment_2_4.ipynb").write_text(json.dumps(nb(cells_24), ensure_ascii=False, indent=1), encoding="utf-8")

    # --- experiment_2_5 ---
    exp25 = r"""
Xr, yr = load_triazines()
oracle = make_regression_oracle(Xr, yr, standard_regcoef(Xr.shape[0]))
n = Xr.shape[1]
x0 = np.zeros(n)
line = {"method": "Wolfe", "alpha_0": 1.0}


def avg_profile(name, inner):
    ot = 0.0
    wt = 0.0
    reps = 5
    for _ in range(reps):
        o = TimedOracle(oracle)
        t0 = time.perf_counter()
        inner(o)
        wt += time.perf_counter() - t0
        ot += o.t_func + o.t_grad + o.t_hess_vec
    ot /= reps
    wt /= reps
    return name, ot, max(wt - ot, 0.0)


rows = [
    avg_profile(
        "L-BFGS",
        lambda o: lbfgs(
            o,
            x0,
            tolerance=1e-2,
            max_iter=8,
            memory_size=10,
            line_search_options=line,
            trace=False,
        ),
    ),
    avg_profile(
        "NLCG",
        lambda o: nonlinear_conjugate_gradients(
            o,
            x0,
            tolerance=1e-2,
            max_iter=8,
            line_search_options=line,
            trace=False,
        ),
    ),
    avg_profile(
        "HFN",
        lambda o: hessian_free_newton(
            o,
            x0,
            tolerance=1e-2,
            max_iter=8,
            line_search_options=line,
            trace=False,
        ),
    ),
]
names = [r[0] for r in rows]
ora = [r[1] for r in rows]
rest = [max(r[2], 0.0) for r in rows]
fig, ax = plt.subplots(figsize=(7, 4))
xpos = np.arange(len(names))
ax.bar(xpos, ora, label="Оракул (func+grad+hess_vec)")
ax.bar(xpos, rest, bottom=ora, label="Прочее (алгебра + лин. поиск + накладные)")
ax.set_xticks(xpos)
ax.set_xticklabels(names)
ax.set_ylabel("Среднее время одной внешней итерации, с")
ax.set_title("П. 2.5: triazines, среднее по 5 запускам (8 внешних итераций)")
ax.legend()
fig.tight_layout()
savefig_both(fig, "exp25_micro_profile")
plt.show()
"""
    cells_25 = [
        md("# Эксперимент 2.5", "Микропрофилирование (`лаб2.pdf`, п. 2.5)."),
        code(AUTORELOAD),
        code(SETUP + exp25),
    ]
    (nbdir / "experiment_2_5.ipynb").write_text(json.dumps(nb(cells_25), ensure_ascii=False, indent=1), encoding="utf-8")

    # --- experiment_2_6 ---
    exp26 = r"""
X, y = load_triazines()
otr, Xte, yte = train_test_oracles_regression(X, y, standard_regcoef(X.shape[0]))
mvAx_te, _, _ = sparse_oracle_ops(Xte)
x0 = np.zeros(X.shape[1])
line = {"method": "Wolfe", "alpha_0": 1.0}
_, msg, h = lbfgs(
    otr,
    x0,
    tolerance=1e-8,
    max_iter=5000,
    memory_size=20,
    line_search_options=line,
    trace=True,
    store_xk=True,
)
assert h and "xk" in h
mse_test = []
for xk in h["xk"]:
    pred = mvAx_te(xk)
    mse_test.append(float(np.mean((yte - pred) ** 2)))
fig, ax1 = plt.subplots(figsize=(9, 4.5))
it = np.arange(len(h["func"]))
ax1.semilogy(it, h["func"], color="C0", label=r"$f_{\mathrm{train}}$")
ax1.semilogy(it, np.array(h["grad_norm"]) ** 2, color="C2", linestyle="--", label=r"$\|\nabla f\|^2$")
ax1.set_xlabel("Итерация k")
ax1.set_ylabel("лог (левая ось)")
ax2 = ax1.twinx()
ax2.plot(it[: len(mse_test)], mse_test, color="C1", lw=2.2, label="MSE на тесте")
ax2.set_ylabel("MSE на тесте (правая ось)")
ax1.set_title("П. 2.6: triazines, train/test 80/20, L-BFGS до ε=10⁻⁸ ({})".format(msg))
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right")
fig.tight_layout()
savefig_both(fig, "exp26_triazines_train_test")
plt.show()
"""
    cells_26 = [
        md("# Эксперимент 2.6", "Оптимизационная точность vs качество на тесте (`лаб2.pdf`, п. 2.6)."),
        code(AUTORELOAD),
        code(SETUP + exp26),
    ]
    (nbdir / "experiment_2_6.ipynb").write_text(json.dumps(nb(cells_26), ensure_ascii=False, indent=1), encoding="utf-8")

    # --- experiment_track1 ---
    exp_t1 = r"""
be = NonConvexOracle()
x0 = np.array([-1.0, 0.5])
line = {"method": "Wolfe", "alpha_0": 1.0}

_, _, h1 = hessian_free_newton(be, x0, tolerance=1e-6, line_search_options=line, trace=True)
_, _, h2 = trust_region_steihaug_newton(be, x0, tolerance=1e-6, trace=True, delta_0=1.0)

fig, ax = plt.subplots(figsize=(6, 5))
gx = np.linspace(-4.5, 4.5, 120)
gy = np.linspace(-2.0, 3.0, 120)
Gx, Gy = np.meshgrid(gx, gy)
Z = beale_mesh_Z(Gx, Gy)
ax.contour(Gx, Gy, Z, levels=25, colors="0.7", linewidths=0.6, alpha=0.85)
xs1 = np.array(h1["x"])
xs2 = np.array(h2["x"])
if xs1.size:
    ax.plot(xs1[:, 0], xs1[:, 1], "o-", label="Усечённый Ньютон (HFN)")
if xs2.size:
    ax.plot(xs2[:, 0], xs2[:, 1], "s-", label="Steihaug–Toint TR")
ax.scatter([x0[0]], [x0[1]], c="k", zorder=5, label="старт")
ax.scatter([3.0], [0.5], c="g", zorder=5, label="минимум")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Трек 1: траектории на функции Била")
ax.legend()
fig.tight_layout()
savefig_both(fig, "exp_t1_beale_traj")
plt.show()

if h2 and "delta" in h2:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(len(h2["delta"])), h2["delta"], "o-")
    ax.set_xlabel("Внешняя итерация k")
    ax.set_ylabel(r"$\Delta_k$")
    ax.set_title("Трек 1: радиус доверительной области")
    fig.tight_layout()
    savefig_both(fig, "exp_t1_delta")
    plt.show()

Xr, yr = load_triazines()
oracle0 = make_regression_oracle(Xr, yr, 0.0)
x0r = np.zeros(Xr.shape[1])
x_opt, msg_tr, _ = trust_region_steihaug_newton(
    oracle0, x0r, tolerance=1e-4, max_iter=200, trace=False
)
print("Track1 triazines lambda=0:", msg_tr, "f=", float(oracle0.func(x_opt)))

x0b = np.array([-1.0, 0.5])
_, m_lm, h_lm = newton_modified(
    be, x0b, tolerance=1e-8, line_search_options=line, trace=True, hessian_mod="lm"
)
_, m_sp, h_sp = newton_modified(
    be,
    x0b,
    tolerance=1e-8,
    line_search_options=line,
    trace=True,
    hessian_mod="spectral",
    spectral_abs=False,
)
fig, ax = plt.subplots(figsize=(7, 4))
if h_lm:
    ax.semilogy(h_lm["time"], h_lm["func"], label="LM-mod ({})".format(m_lm))
if h_sp:
    ax.semilogy(h_sp["time"], h_sp["func"], label="Spectral clip ({})".format(m_sp))
ax.set_xlabel("Время, с")
ax.set_ylabel("f")
ax.set_title("Трек 1: модификации гессиана (Бил)")
ax.legend()
fig.tight_layout()
savefig_both(fig, "exp_t1_hessian_mod_beale")
plt.show()
"""
    cells_t1 = [
        md("# Трек 1", "Steihaug–Toint, доверительная область, ML без L2, модификации гессиана (`лаб2.pdf`, прил. А)."),
        code(AUTORELOAD),
        code(SETUP + exp_t1),
    ]
    (nbdir / "experiment_track1.ipynb").write_text(json.dumps(nb(cells_t1), ensure_ascii=False, indent=1), encoding="utf-8")

    print("Written:", nbdir / "experiment_2_2.ipynb", "… track1")


if __name__ == "__main__":
    main()
