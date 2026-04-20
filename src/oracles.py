import numpy as np
import scipy
from scipy.special import expit

_LOG2 = np.log(2.0)
_EXP_CLIP = 709.0


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """

    def func(self, x):
        raise NotImplementedError("Func oracle is not implemented.")

    def grad(self, x):
        raise NotImplementedError("Grad oracle is not implemented.")

    def hess(self, x):
        raise NotImplementedError("Hessian oracle is not implemented.")

    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError("A should be a symmetric matrix.")
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class NonConvexOracle(BaseSmoothOracle):
    """
    Функция Била (2D). f* = 0 в (3, 0.5).
    """

    def __init__(self):
        pass

    def func(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        xv, yv = x[0], x[1]
        u = 1.5 - xv + xv * yv
        v = 2.25 - xv + xv * yv * yv
        w = 2.625 - xv + xv * yv ** 3
        return u * u + v * v + w * w

    def grad(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        xv, yv = x[0], x[1]
        u = 1.5 - xv + xv * yv
        v = 2.25 - xv + xv * yv * yv
        w = 2.625 - xv + xv * yv ** 3
        ux, uy = yv - 1.0, xv
        vx, vy = yv * yv - 1.0, 2.0 * xv * yv
        wx, wy = yv ** 3 - 1.0, 3.0 * xv * yv * yv
        gx = 2.0 * (u * ux + v * vx + w * wx)
        gy = 2.0 * (u * uy + v * vy + w * wy)
        return np.array([gx, gy], dtype=float)

    def hess(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)
        xv, yv = x[0], x[1]
        u = 1.5 - xv + xv * yv
        v = 2.25 - xv + xv * yv * yv
        w = 2.625 - xv + xv * yv ** 3
        ux, uy = yv - 1.0, xv
        vx, vy = yv * yv - 1.0, 2.0 * xv * yv
        wx, wy = yv ** 3 - 1.0, 3.0 * xv * yv * yv
        uxy, vxy, wxy = 1.0, 2.0 * yv, 3.0 * yv * yv
        uyy, vyy, wyy = 0.0, 2.0 * xv, 6.0 * xv * yv
        hxx = 2.0 * (ux * ux + vx * vx + wx * wx)
        hyy = 2.0 * (
            uy * uy + u * uyy + vy * vy + v * vyy + wy * wy + w * wyy
        )
        hxy = 2.0 * (
            uy * ux + u * uxy + vy * vx + v * vxy + wy * wx + w * wxy
        )
        return np.array([[hxx, hxy], [hxy, hyy]], dtype=float)

    def hess_vec(self, x, v):
        return self.hess(x).dot(np.asarray(v, dtype=float).reshape(-1))


def beale_mesh_Z(X, Y):
    """Значения функции Била на сетке (векторизовано), для контуров."""
    xv = np.asarray(X, dtype=float)
    yv = np.asarray(Y, dtype=float)
    u = 1.5 - xv + xv * yv
    v = 2.25 - xv + xv * yv * yv
    w = 2.625 - xv + xv * yv ** 3
    return u * u + v * v + w * w


class PseudoHuberL2Oracle(BaseSmoothOracle):
    """
    Пакет 3: регрессия, Pseudo-Huber + L2.
    f(x) = (1/m) Σ_i sqrt(1 + ((z_i - y_i)/δ)²) + (λ/2)||x||², z = Ax.
    """

    def __init__(self, matvec_Ax, matvec_ATx, _unused_hess_builder, b, regcoef, delta=1.0):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = np.asarray(b, dtype=float).ravel()
        self.regcoef = float(regcoef)
        self.delta = float(delta)
        if self.delta <= 0.0:
            raise ValueError("delta must be positive")
        self._m = max(int(self.b.size), 1)

    def func(self, x):
        r = self.matvec_Ax(x) - self.b
        u = r / self.delta
        return np.mean(np.sqrt(1.0 + u * u)) + (self.regcoef / 2.0) * np.dot(x, x)

    def grad(self, x):
        r = self.matvec_Ax(x) - self.b
        d2 = self.delta ** 2
        s = np.sqrt(1.0 + (r / self.delta) ** 2)
        w = r / (self._m * d2 * s)
        return self.matvec_ATx(w) + self.regcoef * x

    def hess(self, x):
        raise NotImplementedError(
            "Explicit Hessian is disabled for lab2 ML oracles; use hess_vec(x, v)."
        )

    def hess_vec(self, x, v):
        x = np.asarray(x, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()
        r = self.matvec_Ax(x) - self.b
        d2 = self.delta ** 2
        s2 = 1.0 + (r / self.delta) ** 2
        s = np.sqrt(s2)
        w = 1.0 / (self._m * d2 * (s * s2))
        Av = self.matvec_Ax(v)
        return self.matvec_ATx(w * Av) + self.regcoef * v


class SmoothedSVML2Oracle(BaseSmoothOracle):
    """
    Пакет 3: классификация, smoothed hinge (soft-margin) + L2.
    L(y, z) = ln(1 + exp(1 - yz)), y ∈ {−1, +1}.
    """

    def __init__(self, matvec_Ax, matvec_ATx, _unused_hess_builder, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.b = np.asarray(b, dtype=float).ravel()
        self.regcoef = float(regcoef)
        self._m = max(int(self.b.size), 1)

    def func(self, x):
        Ax = self.matvec_Ax(x)
        t = 1.0 - self.b * Ax
        t = np.clip(t, -_EXP_CLIP, _EXP_CLIP)
        return np.mean(np.log1p(np.exp(t))) + (self.regcoef / 2.0) * np.dot(x, x)

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        t = 1.0 - self.b * Ax
        t = np.clip(t, -_EXP_CLIP, _EXP_CLIP)
        sig = expit(t)
        w = -(self.b * sig) / self._m
        return self.matvec_ATx(w) + self.regcoef * x

    def hess(self, x):
        raise NotImplementedError(
            "Explicit Hessian is disabled for lab2 ML oracles; use hess_vec(x, v)."
        )

    def hess_vec(self, x, v):
        x = np.asarray(x, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()
        Ax = self.matvec_Ax(x)
        t = 1.0 - self.b * Ax
        t = np.clip(t, -_EXP_CLIP, _EXP_CLIP)
        sig = expit(t)
        d = (sig * (1.0 - sig)) / self._m
        Av = self.matvec_Ax(v)
        return self.matvec_ATx(d * Av) + self.regcoef * v


# Совместимость с ожидаемыми именами классов в остальном коде
REG_MODEL_NAMEL2Oracle = PseudoHuberL2Oracle
CLASS_MODEL_NAMEL2Oracle = SmoothedSVML2Oracle


def hess_vec_finite_diff(func, x, v, eps=None):
    """
    Аппроксимация (∇²f(x) v)_i по п. 1.4 методички (лаб. 2).
    """
    x = np.asarray(x, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    n = x.size
    if eps is None:
        eps = np.cbrt(np.finfo(float).eps)
    out = np.zeros(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        num = (
            func(x + eps * v + eps * ei)
            - func(x + eps * v)
            - func(x + eps * ei)
            + func(x)
        )
        out[i] = num / (eps * eps)
    return out
