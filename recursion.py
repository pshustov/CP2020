import sympy
import math
x = sympy.Symbol('x')
N = 25
exact = [float(sympy.integrate(x**n * sympy.exp(1 - x), (x, 0, 1))) for n in range(N)]
def upwards_recursion(n):
    values = []
    values.append(math.exp(1) - 1)
    i = 1
    while i < n:
        I_previous = values[-1]
        I_actual = I_previous*i - 1
        values.append(I_actual)
        i += 1
    return values


values = upwards_recursion(25)

from numpy.testing import assert_allclose
assert_allclose(values, exact)

def downwards_recursion(n):
    values = []
    values.append(0)
    i = n-1
    while i > 0:
        I_actual = values[-1]
        I_previous = (I_actual + 1)/i
        values.append(I_previous)
        i -= 1
    return list(reversed(values))


values = downwards_recursion(25)

from numpy.testing import assert_allclose
assert_allclose(values, exact)