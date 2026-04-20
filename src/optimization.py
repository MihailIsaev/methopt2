import time
from collections import defaultdict, deque

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import cho_factor, cho_solve

from utils import get_line_search_tool


def _rel_grad_sq_tol(oracle, x, x0, tolerance):
    g = oracle.grad(x)
    g0 = oracle.grad(x0)
    return np.dot(g, g) <= tolerance * np.dot(g0, g0)


def _append_outer_history(history, trace, oracle, x, t0):
    if not trace:
        return
    history["time"].append(time.perf_counter() - t0)
    history["func"].append(float(oracle.func(x)))
    history["grad_norm"].append(float(np.linalg.norm(oracle.grad(x))))
    if x.size <= 2:
        history["x"].append(np.copy(x))


def _steihaug_boundary_tau(d, p, delta, alpha_cap=None):
    """Положительный корень ||d + tau p|| = delta; при alpha_cap — ищем в (0, alpha_cap]."""
    a = float(np.dot(p, p))
    if a < 1e-30:
        return None
    b = 2.0 * float(np.dot(d, p))
    c = float(np.dot(d, d)) - float(delta) ** 2
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        disc = 0.0
    sqrt_disc = np.sqrt(disc)
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    pos = [t for t in roots if t > 1e-14]
    if not pos:
        return None
    if alpha_cap is not None:
        pos = [t for t in pos if t <= alpha_cap + 1e-10]
        if not pos:
            return None
    return min(pos)


def steihaug_toint_subproblem(oracle, x, delta, eta, g=None, max_inner=500):
    """
    Внутренняя подзадача доверительной области (методичка, трек 1, прилож. А).
    Модель: m(d) = g^T d + 1/2 d^T H d, g = ∇f(x), H = ∇²f(x).
    Возвращает (d, info) с полями n_inner, reason.
    """
    if g is None:
        g = oracle.grad(x)
    g = np.asarray(g, dtype=float).ravel()
    g_norm = np.linalg.norm(g)
    eta_use = max(float(eta), 1e-12)
    d = np.zeros_like(g)
    r = np.copy(g)
    p = -r
    n_inner = 0
    for _ in range(max_inner):
        Hp = oracle.hess_vec(x, p)
        kappa = float(np.dot(p, Hp))
        n_inner += 1

        if kappa <= 1e-18:
            tau = _steihaug_boundary_tau(d, p, delta, None)
            if tau is None:
                return d, {"n_inner": n_inner, "reason": "curvature_no_intersection"}
            return d + tau * p, {"n_inner": n_inner, "reason": "negative_curvature"}

        alpha = float(np.dot(r, r)) / kappa
        step = d + alpha * p
        if np.linalg.norm(step) >= delta - 1e-14:
            tau = _steihaug_boundary_tau(d, p, delta, alpha)
            if tau is None:
                return d, {"n_inner": n_inner, "reason": "boundary_fail"}
            return d + tau * p, {"n_inner": n_inner, "reason": "trust_boundary"}

        d = step
        rs = float(np.dot(r, r))
        r = r + alpha * Hp
        if g_norm > 0 and np.linalg.norm(r) <= eta_use * g_norm:
            return d, {"n_inner": n_inner, "reason": "tol"}

        rs_new = float(np.dot(r, r))
        if rs < 1e-40:
            return d, {"n_inner": n_inner, "reason": "small_rs"}
        beta = rs_new / rs
        p = -r + beta * p

    return d, {"n_inner": n_inner, "reason": "max_inner"}


def trust_region_steihaug_newton(
    oracle,
    x_0,
    tolerance=1e-4,
    max_iter=500,
    trace=False,
    display=False,
    delta_0=1.0,
):
    """
    Усечённый Ньютон в доверительной области (Steihaug–Toint + обновление Δ).
    """
    history = defaultdict(list) if trace else None
    t0 = time.perf_counter()
    x = np.asarray(x_0, dtype=float).ravel().copy()
    delta = float(delta_0)
    _append_outer_history(history, trace, oracle, x, t0)

    for _ in range(max_iter):
        if _rel_grad_sq_tol(oracle, x, x_0, tolerance):
            return x, "success", history

        g = oracle.grad(x)
        g_norm = np.linalg.norm(g)
        eta_k = min(0.5, np.sqrt(max(g_norm, 1e-30)))

        d, _info = steihaug_toint_subproblem(oracle, x, delta, eta_k, g=g)

        f0 = float(oracle.func(x))
        Hd = oracle.hess_vec(x, d)
        pred = -(float(np.dot(g, d)) + 0.5 * float(np.dot(d, Hd)))
        if pred <= 1e-30:
            pred = 1e-30

        x_new = x + d
        f1 = float(oracle.func(x_new))
        rho = (f0 - f1) / pred

        if trace:
            history["delta"].append(delta)

        if rho <= 0.0:
            delta = 0.25 * delta
            x_new = x
        else:
            x = x_new
            if rho < 0.25:
                delta = 0.25 * delta
            elif rho > 0.75 and np.linalg.norm(d) >= delta - 1e-10:
                delta = 2.0 * delta

        _append_outer_history(history, trace, oracle, x, t0)

    return x, "iterations_exceeded", history


def linear_conjugate_gradients(
    matvec,
    b,
    x_0,
    tolerance=1e-4,
    max_iter=None,
    trace=False,
    display=False,
):
    """
    Метод сопряжённых градиентов для Ax = b (одно матвек-умножение за итерацию).
    """
    history = defaultdict(list) if trace else None
    b = np.asarray(b, dtype=float).ravel()
    b_norm = np.linalg.norm(b)
    n = b.size
    if max_iter is None:
        max_iter = n

    x_k = np.asarray(x_0, dtype=float).ravel().copy()
    g_k = matvec(x_k) - b
    d_k = -g_k
    t0 = time.perf_counter()

    def record():
        if not trace:
            return
        history["time"].append(time.perf_counter() - t0)
        history["residual_norm"].append(float(np.linalg.norm(g_k)))
        if x_k.size <= 2:
            history["x"].append(np.copy(x_k))

    record()

    for _ in range(max_iter):
        if b_norm == 0.0:
            if np.linalg.norm(g_k) <= tolerance:
                return x_k, "success", history
        else:
            if np.linalg.norm(g_k) <= tolerance * b_norm:
                return x_k, "success", history

        Ad = matvec(d_k)
        denom = float(np.dot(d_k, Ad))
        if abs(denom) < 1e-30:
            return x_k, "breakdown", history

        alpha = float(np.dot(g_k, g_k)) / denom
        x_k = x_k + alpha * d_k
        g_next = g_k + alpha * Ad

        if b_norm == 0.0:
            if np.linalg.norm(g_next) <= tolerance:
                g_k = g_next
                record()
                return x_k, "success", history
        else:
            if np.linalg.norm(g_next) <= tolerance * b_norm:
                g_k = g_next
                record()
                return x_k, "success", history

        beta = float(np.dot(g_next, g_next)) / float(np.dot(g_k, g_k))
        d_k = -g_next + beta * d_k
        g_k = g_next
        record()

    return x_k, "iterations_exceeded", history


def _truncated_newton_direction(oracle, x_k, g_k, eta_k, max_inner, trace_inner):
    """Решает H d ≈ -g внутренним CG; старт x_0 = -g (рекомендация методички)."""

    def matvec(v):
        return oracle.hess_vec(x_k, v)

    b = -g_k
    x0 = -g_k
    eta = max(float(eta_k), 1e-12)

    while True:
        d, msg, _ = linear_conjugate_gradients(
            matvec,
            b,
            x0,
            tolerance=eta,
            max_iter=max_inner,
            trace=False,
            display=False,
        )
        if msg == "breakdown":
            return None, "cg_breakdown"

        gd = float(np.dot(g_k, d))
        if gd < 0.0:
            return d, "ok"

        eta = eta * 0.1
        if eta < 1e-12:
            return None, "no_descent"
        x0 = -g_k


def nonlinear_conjugate_gradients(
    oracle,
    x_0,
    tolerance=1e-4,
    max_iter=500,
    line_search_options=None,
    display=False,
    trace=False,
):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x = np.asarray(x_0, dtype=float).ravel().copy()
    t0 = time.perf_counter()
    _append_outer_history(history, trace, oracle, x, t0)

    g_prev = None
    d_prev = None
    prev_alpha = None

    for _ in range(max_iter):
        if _rel_grad_sq_tol(oracle, x, x_0, tolerance):
            return x, "success", history

        g = oracle.grad(x)
        if g_prev is None:
            d = -g
        else:
            y = g - g_prev
            beta = float(np.dot(g, y)) / float(np.dot(g_prev, g_prev) + 1e-30)
            if beta < 0.0:
                beta = 0.0
            d = -g + beta * d_prev
            if float(np.dot(d, g)) >= 0.0:
                d = -g

        alpha = line_search_tool.line_search(oracle, x, d, prev_alpha)
        if alpha is None or not np.isfinite(alpha):
            return x, "computational_error", history

        prev_alpha = alpha
        x = x + alpha * d
        g_prev = np.copy(g)
        d_prev = np.copy(d)
        _append_outer_history(history, trace, oracle, x, t0)

    return x, "iterations_exceeded", history


def lbfgs(
    oracle,
    x_0,
    tolerance=1e-4,
    max_iter=500,
    memory_size=10,
    line_search_options=None,
    display=False,
    trace=False,
    store_xk=False,
):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x = np.asarray(x_0, dtype=float).ravel().copy()
    t0 = time.perf_counter()
    _append_outer_history(history, trace, oracle, x, t0)
    if trace and store_xk:
        history.setdefault("xk", []).append(np.copy(x))

    s_hist = deque(maxlen=max(memory_size, 1))
    y_hist = deque(maxlen=max(memory_size, 1))

    def two_loop(g):
        if memory_size == 0 or len(s_hist) == 0:
            return -g
        pairs = list(zip(s_hist, y_hist))
        mloc = len(pairs)
        q = np.array(g, dtype=float).copy()
        alphas = np.empty(mloc, dtype=float)
        for i in range(mloc - 1, -1, -1):
            s, y = pairs[i]
            ys = float(np.dot(y, s))
            rho = 1.0 / ys
            alphas[i] = rho * float(np.dot(s, q))
            q = q - alphas[i] * y
        s_last, y_last = pairs[-1]
        yy = float(np.dot(y_last, y_last))
        ys = float(np.dot(y_last, s_last))
        gamma0 = ys / (yy + 1e-30) if yy > 1e-30 else 1.0
        z = gamma0 * q
        for i in range(mloc):
            s, y = pairs[i]
            ys = float(np.dot(y, s))
            rho = 1.0 / ys
            beta = rho * float(np.dot(y, z))
            z = z + s * (alphas[i] - beta)
        return -z

    g_old = oracle.grad(x)
    prev_alpha = None

    for _ in range(max_iter):
        if _rel_grad_sq_tol(oracle, x, x_0, tolerance):
            return x, "success", history

        d = two_loop(g_old)
        alpha = line_search_tool.line_search(oracle, x, d, prev_alpha)
        if alpha is None or not np.isfinite(alpha):
            return x, "computational_error", history
        prev_alpha = alpha

        s = alpha * d
        x_new = x + s
        g_new = oracle.grad(x_new)
        yv = g_new - g_old

        if memory_size > 0:
            ys = float(np.dot(yv, s))
            if ys > 1e-14:
                s_hist.append(s.copy())
                y_hist.append(yv.copy())

        x = x_new
        g_old = g_new
        _append_outer_history(history, trace, oracle, x, t0)
        if trace and store_xk:
            history.setdefault("xk", []).append(np.copy(x))

    return x, "iterations_exceeded", history


def hessian_free_newton(
    oracle,
    x_0,
    tolerance=1e-4,
    max_iter=500,
    line_search_options=None,
    display=False,
    trace=False,
    cg_max_iter=None,
):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x = np.asarray(x_0, dtype=float).ravel().copy()
    t0 = time.perf_counter()
    _append_outer_history(history, trace, oracle, x, t0)
    n = x.size
    if cg_max_iter is None:
        cg_max_iter = n

    for _ in range(max_iter):
        if _rel_grad_sq_tol(oracle, x, x_0, tolerance):
            return x, "success", history

        g = oracle.grad(x)
        g_norm = np.linalg.norm(g)
        eta_k = min(0.5, np.sqrt(max(g_norm, 1e-30)))

        d, inner_msg = _truncated_newton_direction(
            oracle, x, g, eta_k, max_inner=cg_max_iter, trace_inner=False
        )
        if d is None:
            return x, inner_msg, history

        alpha = line_search_tool.line_search(oracle, x, d, None)
        if alpha is None or not np.isfinite(alpha):
            return x, "computational_error", history
        x = x + alpha * d
        _append_outer_history(history, trace, oracle, x, t0)

    return x, "iterations_exceeded", history


def gradient_descent(
    oracle,
    x_0,
    tolerance=1e-4,
    max_iter=10000,
    line_search_options=None,
    trace=False,
    display=False,
):
    history = defaultdict(list) if trace else None
    ls = get_line_search_tool(line_search_options)
    x = np.asarray(x_0, dtype=float).ravel().copy()
    t0 = time.perf_counter()
    _append_outer_history(history, trace, oracle, x, t0)
    prev_alpha = None
    for _ in range(max_iter):
        if _rel_grad_sq_tol(oracle, x, x_0, tolerance):
            return x, "success", history
        d = -oracle.grad(x)
        alpha = ls.line_search(oracle, x, d, prev_alpha)
        if alpha is None or not np.isfinite(alpha):
            return x, "computational_error", history
        prev_alpha = alpha
        x = x + alpha * d
        _append_outer_history(history, trace, oracle, x, t0)
    return x, "iterations_exceeded", history


def newton(
    oracle,
    x_0,
    tolerance=1e-4,
    max_iter=100,
    line_search_options=None,
    trace=False,
    display=False,
):
    history = defaultdict(list) if trace else None
    ls = get_line_search_tool(line_search_options)
    x = np.asarray(x_0, dtype=float).ravel().copy()
    t0 = time.perf_counter()
    _append_outer_history(history, trace, oracle, x, t0)
    for _ in range(max_iter):
        if _rel_grad_sq_tol(oracle, x, x_0, tolerance):
            return x, "success", history
        g = oracle.grad(x)
        try:
            H = oracle.hess(x)
            c, low = cho_factor(H, lower=True)
            d = cho_solve((c, low), -g)
        except LinAlgError:
            return x, "newton_direction_error", history
        alpha = ls.line_search(oracle, x, d, None)
        if alpha is None or not np.isfinite(alpha):
            return x, "computational_error", history
        x = x + alpha * d
        _append_outer_history(history, trace, oracle, x, t0)
    return x, "iterations_exceeded", history


def newton_modified(
    oracle,
    x_0,
    tolerance=1e-4,
    max_iter=100,
    line_search_options=None,
    trace=False,
    display=False,
    hessian_mod="lm",
    lm_gamma0=1e-5,
    lm_gamma_max=1e15,
    spectral_eps=1e-5,
    spectral_abs=False,
    max_lm_tries=40,
):
    """
    Ньютон с модификацией гессиана (LM / спектральная) — для сравнения в треке.
    """
    from scipy.linalg import eigh

    history = defaultdict(list) if trace else None
    ls = get_line_search_tool(line_search_options)
    x = np.asarray(x_0, dtype=float).ravel().copy()
    n = x.size
    I = np.eye(n)
    t0 = time.perf_counter()
    _append_outer_history(history, trace, oracle, x, t0)
    prev_alpha = None
    gamma = float(lm_gamma0)

    for _ in range(max_iter):
        if _rel_grad_sq_tol(oracle, x, x_0, tolerance):
            return x, "success", history

        g = oracle.grad(x)
        H = np.asarray(oracle.hess(x), dtype=float)
        f_k = float(oracle.func(x))

        if hessian_mod == "spectral":
            lam, V = eigh(H)
            if spectral_abs:
                lam_mod = np.maximum(np.abs(lam), spectral_eps)
            else:
                lam_mod = np.maximum(lam, spectral_eps)
            H_bar = (V * lam_mod[np.newaxis, :]) @ V.T
            try:
                c, low = cho_factor(H_bar, lower=True)
                d = cho_solve((c, low), -g)
            except LinAlgError:
                return x, "newton_direction_error", history
            alpha = ls.line_search(oracle, x, d, prev_alpha)
            if alpha is None or not np.isfinite(alpha):
                return x, "computational_error", history
            prev_alpha = alpha
            x = x + alpha * d
            if trace:
                history["gamma"].append(np.nan)
        elif hessian_mod == "lm":
            accepted = False
            for _t in range(max_lm_tries):
                H_bar = H + gamma * I
                try:
                    c, low = cho_factor(H_bar, lower=True)
                    d_try = cho_solve((c, low), -g)
                except LinAlgError:
                    gamma = min(gamma * 10.0, lm_gamma_max)
                    continue
                alpha = ls.line_search(oracle, x, d_try, prev_alpha)
                if alpha is None or not np.isfinite(alpha):
                    gamma = min(gamma * 10.0, lm_gamma_max)
                    continue
                x_try = x + alpha * d_try
                try:
                    f_try = float(oracle.func(x_try))
                except Exception:
                    gamma = min(gamma * 10.0, lm_gamma_max)
                    continue
                if f_try >= f_k - 1e-18:
                    gamma = min(gamma * 10.0, lm_gamma_max)
                    continue
                x = x_try
                prev_alpha = alpha
                gamma = max(gamma / 10.0, 1e-20)
                accepted = True
                if trace:
                    history["gamma"].append(gamma)
                break
            if not accepted:
                if gamma >= lm_gamma_max * 0.99:
                    return x, "newton_direction_error", history
                return x, "computational_error", history
        else:
            raise ValueError("Unknown hessian_mod {!r}".format(hessian_mod))

        _append_outer_history(history, trace, oracle, x, t0)

    return x, "iterations_exceeded", history
