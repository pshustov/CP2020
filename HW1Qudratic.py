import cmath
from numpy import allclose


def solve_quad(b, c):
    D =  cmath.sqrt((b**2) - (4*c))
    if b > 0:
        x1 = (-b - D) / 2
        x2 = c/x1
    else:
        x1 = (-b + D) / 2
        x2 = c/x1
    return x1,x2


variants = [{'b': 4.0, 'c': 3.0},
            {'b': 2.0, 'c': 1.0},
            {'b': 0.5, 'c': 4.0},
            {'b': 1e10, 'c': 3.0},
            {'b': -1e10, 'c': 4.0},]

for var in variants:
    x1, x2 = solve_quad(**var)
    print(allclose(x1*x2, var['c']))

