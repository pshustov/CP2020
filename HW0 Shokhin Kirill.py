import sympy
import numpy as np
from numpy.testing import assert_allclose
x = sympy.Symbol('x')
N = 25
exact = np.asarray([float(sympy.integrate(x**i * sympy.exp(1 - x), (x, 0, 1))) for i in range(N)])
H = np.zeros(N)
G = np.zeros(N)
H[0] = np.e-1
G[N-1] = exact[N-1]


def upwards_recursion(n):
    for i in range(n-1):
        H[i+1] = (i+1) * H[i] - 1
    return H
values_up = upwards_recursion(N)
assert_allclose(values_up, exact)
"""Видно, что последние 6 элементов начинают сильно расходиться"""


def downwards_recursion(n):
    for i in range(n-1, 0, -1):
        G[i-1] = (1+G[i])/i
    return G


values_down = downwards_recursion(N)
assert_allclose(values_down, exact)
"""А в обратном случае рекурсия полностью совпадает с численным методом"""