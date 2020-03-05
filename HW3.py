import numpy as np
import matplotlib.pyplot as plt
import math

"""Part 1"""
def newton_iteration(f, fder, x0, eps=1e-5, maxiter=1000, m=1):
    x = x0
    for niter in range(1, maxiter):
        y = x
        x = y - m * f(y) / fder(y)
        # print(x, niter)
        if abs(x - y) / m < eps:
            break
    return x, niter, m


def f(x):
    return x**2-1
def fder(x):
    return 2*x


for i in range(10,200,50):
    print('для начальной точки:', i, newton_iteration(f, fder, i, m=1))
    print('для начальной точки:', -i, newton_iteration(f, fder, -i, m=1), "\n")


def f1(x):
    return (x**2-1)**2
def fder1(x):
    return 4*x*(x**2-1)


for i in range(10,200,50):
    for j in range(1, 6):
        print('для начальной точки:', i, newton_iteration(f1, fder1, i, m=j))
        print('для начальной точки:', -i, newton_iteration(f1, fder1, -i, m=j), "\n")
# Видно, что корней кратности 1 значения m!=1 не работают. Для корней кратности 2, самый оптимальный вариант m=2, который дает квадратичную сходимость.


"""Part 2"""
x = np.linspace(0, 1.4, 1000)
# plt.plot(x, np.sqrt(x))
# plt.plot(x, np.cos(x))
plt.grid(True)



def fixed_point(x0, eps=1e-5, maxiter=1000):
    y = x0
    for niter in range(maxiter):
        x = math.cos(y)**2
        if abs(x-y) < eps:
            break
        y = x
    return x, niter

print(fixed_point(-100))


def mod_fixed_point(x0, a, eps=1e-5, maxiter=1000):
    y = x0
    for niter in range(maxiter):
        x = y - a*(y-math.cos(y)**2)
        if abs(x-y) < eps:
            break
        y = x
    return x, niter


a = np.linspace(0, 1, 1000)
plt.plot([i for i in a], [mod_fixed_point(-1, i)[1] for i in a])
#График зависимости кол-ва итераций от параметра альфа, где при aльфа>1 нет решений.
a1 = 2/(1+math.sin(2*0.7)+1+math.sin(2*0.6))
plt.plot(a1, mod_fixed_point(-1, a1)[1], 'o')
# Найшли критическую альфа, для минимума итераций.
plt.show()