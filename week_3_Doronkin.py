from random import random
import numpy as np
import matplotlib.pyplot as plt







def newton_iteration(f, fder, x0, eps=1e-5, maxiter=1000):

    """Find a root of $f(x) = 0$ via Newton's iteration starting from x0.

    Parameters
    ----------
    f : callable
        The function to find a root of.
    fder : callable
        The derivative of `f`.
    x0 : float
        Initial value for the Newton's iteration.
    eps : float
        The target accuracy.
        The iteration stops when the distance between successive iterates is below `eps`.
        Default is 1e-5.
    maxiter : int
        The maximum number of iterations (default is 1000.)
        Iterations terminate if the number of iterations exceeds `maxiter`.
        This parameter is only needed to avoid infinite loops if iterations wander off.

    Returns
    -------
    x : float
        The estimate for the root.
    niter : int
        The number of iterations.
    """

    niter = 0
    x = x0 + 2 * eps

    for niter in range(maxiter):
        x = x0 - f(x0) / fder(x0)
        res = abs(x - x0)
        if res <= eps:
            break
        x0 = x
    return x, niter+1









#первый тест

def f(x):
    result = x**2 - 1
    return result

def fder(x):
    result = 2 * x
    return result
print('выведем результат первого теста:')
print(newton_iteration(f, fder, random() * 50)) #вобьем рандомные x0, параметры eps и maxiter оставляем по умолчанию










#для второго теста необходимо модифицировать newton iteration, добавив множественные корни

def modified_newton_iteration(f, fder, x0, m, eps=1e-5, maxiter=1000):
    niter = 0
    x = x0 + 2 * eps
    pts = []
    for niter in range(maxiter):
        x = x0 - (m * f(x0)) / fder(x0)
        pts.append(x)
        res = abs(x - x0)
        if res <= eps:
            break
        x0 = x
    return x, niter+1, pts

def f1(x):
    result = (x**2 - 1)**2
    return result
def fder1(x):
    result = 4 * x ** 3 - 4 * x
    return result


print('выведем результаты второго теста:')

for m in range(1, 6):
    print(modified_newton_iteration(f1, fder1, random() * 2.5, m))  #так же рандомные x0, и дефолтные eps и maxiter

    #теперь выведем результаты на графики
    x, niter, pts = modified_newton_iteration(f1, fder1, random() * 2.5, m)
    plt.grid()
    plt.plot(pts, np.arange(0, niter))
    plt.show()












#fixed point метод


def vec_norm(dim , vec): #норма вектора

    result = 0
    for i in range(dim):
       result = result + vec[i] * vec[i]
    vector_norm = np.sqrt(result)
    return vector_norm


def fixed_point(f, x0, eps=1e-5, maxiter=1000): #функция fixed point
    """Find a root of $x = f(x)$ via fixed-point iteration algorithm from x0.

    Parameters
    ----------
    f : callable
        The function to find a root of.
    x0 : float
        Initial value for the Newton's iteration.
    eps : float
        The target accuracy.
        The iteration stops when the distance between successive iterates is below `eps`.
        Default is 1e-5.
    maxiter : int
        The maximum number of iterations (default is 1000.)
        Iterations terminate if the number of iterations exceeds `maxiter`.
        This parameter is only needed to avoid infinite loops if iterations wander off.

    Returns
    -------
    x : float
        The estimate for the root.
    niter : int
        The number of iterations.
    """
    niter = 0
    x = 0

    for niter in range(maxiter):

        x = x0 - f(x0)
        res = x0 - x
        res_error = vec_norm(1, res)

        if res_error <= eps:
            break

        x0 = x

    return x, niter




#модифицированный метод fixed point c добавлением параметра a

def fixed_point_another(f, x0, a, eps=1e-5, maxiter=1000):

    niter = 0
    x = x0 - a * f(x0) #формула для x0 с параметром a

    for niter in range(maxiter):

        x = x0 - a * f(x0)
        res = x0 - x
        res_error = vec_norm(1, res)

        if not res_error > eps:
            break
        x0 = x

    return x, niter







