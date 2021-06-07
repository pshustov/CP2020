import numpy as np
from scipy import integrate
from scipy.special import roots_chebyt
from scipy.special import roots_legendre
from scipy.special import roots_sh_legendre
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def f(x):
    return 7*x**3 - 8*x**2 - 2*x + 3




def f_2(x):
    return 5 * (875*x**3 + 2425*x**2 + 2210*x + 663)



def f_3(x):
    return np.cos(2 * np.pi * x) / 2





def midpoint_rule(f, a, b, eps):
    definite_value = abs(integrate.quad(f, a, b)[0]) #считаем точность scipy.integrate.quad достаточной чтобы считать
    # его аналитически рассчитанным (посчитанным руками). Убедиться в этом несложно будет впоследствии.
    print(definite_value)
    iterations = 1
    integral = 0
    while abs(definite_value - integral) > eps:
        step = (b - a) / iterations
        integral = 0
        for i in range(iterations):
            integral += step * f(a + (i - 1) * step)
        iterations *= 2
    print("definite hand-calculated value : {} -- {} : value of the "
          "midpoint integration".format(definite_value, integral))




def g(x):
    return np.sin(np.pi * x)

def exact_solution(x):
    return np.sin(np.pi * x) + 2 / np.pi



def main_1():

    midpoint_rule(f, -1, 1, 1e-3)

    for n in (1, 2, 3, 4, 5, 6):  #в условии просят взять первые 6

        roots, weights = roots_legendre(n)
        print('Estimated: {}, number_of_the_roots: {}'.format(sum(weights * f(roots)), n))



    I_calculated_with_the_estimated_formula = abs(integrate.quad(f_2, -1, 1)[0]) #следует отметить
    # что в действительности без приближения интеграла от f,
    # значение будет равно примерно тому, что ниже (I_REAL), но мы его не используем:

    I_REAL = abs(integrate.quad(f, 0, 10)[0])

    print('Answer with estimation: {}, '
          'definite answer with scipy.integrate: {} '.format(I_calculated_with_the_estimated_formula, I_REAL))

    eps = 1e-10
    n = 1
    I = np.inf
    while eps < abs(I_calculated_with_the_estimated_formula - I):
        roots, weights = roots_legendre(n)
        I = sum(weights * f_2(roots))
        print('Estimated: {}, number_of_the_roots_on_Legendre: {}'.format(I, n))
        n += 1


    eps = 1e-10
    n = 1
    I_0 = np.inf
    I = 0
    checker = True

    while checker == True:
        roots, weights = roots_chebyt(n)
        I = sum(weights * f_3(roots))
        checker = abs(I_0 - I) != 0
        I_0 = I
        n += 1
    print('Estimated: {}, number_of_the_roots_on_Chebyshev: {}'.format(I, n-1))




def main_2():


    eps = 1e-15  #возьмем очень большую точность, чтобы было побольше точек в итоговой аппроксимации
    N = 1 #счетчик
    result_array = []
    error_array = []
    roots_array = []
    error = np.inf

    while error >= eps:
        roots, weights = roots_sh_legendre(N)
        matrix_of_approximation = np.empty([N, N])
        matrix_of_approximation[:] = -0.5 * weights
        numeric_solution = np.linalg.solve(np.eye(N) + matrix_of_approximation, g(roots))
        error = np.linalg.norm(exact_solution(roots) - numeric_solution)
        result_array.append(numeric_solution)
        roots_array.append(roots)
        error_array.append(error)
        N += 1

    x_linspace = np.linspace(min(roots_array[-1]), max(roots_array[-1]), 50)

    for i in range(len(result_array)):
        plt.figure()
        plt.title('Gaussian Quad for Fredholm Equasion')
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x_linspace, exact_solution(x_linspace), 'g', label='Exact solution')
        plt.plot(roots_array[i], result_array[i], 'o', label='Estimated solution using {} points'.format(i+1))
        plt.legend()
        plt.show()

    #блок интерполяции
    plt.figure()
    plt.title('Interpolation')
    interp_func = interp1d(roots_array[8], result_array[8])
    plt.plot(x_linspace, interp_func(x_linspace), '-', label='Interpolation on the {} points'.format(9))
    plt.grid()
    plt.plot(x_linspace, exact_solution(x_linspace), 'g', label='Exact solution')
    #график интерполяции с помощью scipy.interpolate.interp1d

    plt.legend()
    plt.show()



if __name__ == main_1():
    main_1()
if __name__ == main_2():
    main_2()



