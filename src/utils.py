import numpy as np
from scipy.optimize._linesearch import scalar_search_wolfe2


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
            - 'Best' -- optimal step size inferred via analytical minimization.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """

    def __init__(self, method="Wolfe", **kwargs):
        self._method = method
        if self._method == "Wolfe":
            self.c1 = kwargs.get("c1", 1e-4)
            self.c2 = kwargs.get("c2", 0.9)
            self.alpha_0 = kwargs.get("alpha_0", 1.0)
        elif self._method == "Armijo":
            self.c1 = kwargs.get("c1", 1e-4)
            self.alpha_0 = kwargs.get("alpha_0", 1.0)
        elif self._method == "Constant":
            self.c = kwargs.get("c", 1.0)
        elif self._method == "Best":
            pass
        else:
            raise ValueError("Unknown method {}".format(method))

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == "Constant":
            return self.c
        if previous_alpha is not None:
            alpha = previous_alpha
        else:
            alpha = self.alpha_0
        if self._method == "Armijo":
            return self._armijo_search(oracle, x_k, d_k, alpha)
        if self._method == "Wolfe":
            return self._wolfe_search(oracle, x_k, d_k, alpha)
        if self._method == "Best":
            return self._best_search(oracle, x_k, d_k)
        raise ValueError("Unknown method {}".format(self._method))

    def _armijo_search(self, oracle, x_k, d_k, alpha_0):
        alpha = alpha_0
        phi_0 = oracle.func_directional(x_k, d_k, 0)
        grad_phi_0 = oracle.grad_directional(x_k, d_k, 0)

        while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * grad_phi_0:
            alpha /= 2.0
            if alpha < 1e-16:
                return None

        return alpha

    def _wolfe_search(self, oracle, x_k, d_k, alpha_0):
        phi = lambda a: oracle.func_directional(x_k, d_k, a)
        derphi = lambda a: oracle.grad_directional(x_k, d_k, a)

        alpha, *_ = scalar_search_wolfe2(phi, derphi, c1=self.c1, c2=self.c2)
        if alpha is None:
            return self._armijo_search(oracle, x_k, d_k, alpha_0)
        return alpha

    def _best_search(self, oracle, x_k, d_k):
        """
        Exact step for a quadratic model along the ray: minimize
        phi(alpha) ≈ phi(0) + alpha phi'(0) + 0.5 alpha^2 phi''(0),
        phi''(0) = d^T H d when oracle implements hess().
        """
        g = oracle.grad(x_k)
        phi_p = float(np.dot(g, d_k))
        try:
            Hv = oracle.hess_vec(x_k, d_k)
        except Exception:
            try:
                H = oracle.hess(x_k)
                Hv = H.dot(d_k)
            except Exception:
                return None
        curv = float(np.dot(d_k, Hv))
        if curv <= 1e-18:
            return None
        alpha = -phi_p / curv
        if alpha <= 0.0 or not np.isfinite(alpha):
            return None
        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if isinstance(line_search_options, LineSearchTool):
            return line_search_options
        return LineSearchTool.from_dict(line_search_options)
    return LineSearchTool()
