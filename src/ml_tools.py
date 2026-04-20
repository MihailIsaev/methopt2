"""
Операции matvec для разреженной или плотной матрицы признаков A.

В лабе 2 ML-оракулы обязаны работать без явного построения гессиана, поэтому
наружу отдаём только умножения `A x` и `A^T u`. Третий возвращаемый объект
оставлен как заглушка `None` ради совместимости с уже написанными ноутбуками.
"""
from scipy.sparse import csr_matrix


def sparse_oracle_ops(A):
    """
    Parameters
    ----------
    A : scipy.sparse.csr_matrix or ndarray, shape (m, n)
    """
    if not hasattr(A, "tocsr"):
        A = csr_matrix(A)
    else:
        A = A.tocsr()
    m, _ = A.shape

    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(u):
        return A.T.dot(u)

    return matvec_Ax, matvec_ATx, None
