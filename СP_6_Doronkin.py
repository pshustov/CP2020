from sympy import *
from scipy import integrate
import matplotlib.pyplot as plt



def one_sided_deriv(f, x0, h):
    x = symbols('x')
    f1 = f.subs({x: x0})
    f2 = f.subs({x: x0 + h})
    derivative = (f2 - f1) / (h)
    return derivative


def two_point_deriv(f, x0, h):
    x = symbols('x')
    f1 = f.subs({x: x0+h})
    f2 = f.subs({x: x0-h})
    derivative = (f1 - f2) / (2*h)
    return derivative


def three_point_deriv(f, x0, h):
    x = symbols('x')
    if x0 == 0:
        f0 = 0.0
    else:
        f0 = f.subs({x: x0})
    f1 = f.subs({x: x0 + h})
    f2 = f.subs({x: x0 + 2*h})
    derivative = (- 3/2 * f0 + 2 * f1 - 1/2 * f2) / h
    return derivative



def f_test_1(x):
    return x**3

def f_test_2(x):
    return x**2 * log(x)

def der_f_test_2(x):
    if x == 0:
        return 0.0
    else:
        return x * (2.*log(x) + 1)



def midpoint_rule(f, a, b, eps):
    definite_value = abs(integrate.quad(f, a, b)[0]) #считаем точность scipy.integrate.quad достаточной чтобы считать
    # его аналитически рассчитанным (посчитанным руками). Убедиться в этом несложно будет впоследствии.
    iterations = 2
    integral = 0
    errors = []
    niter = []
    while abs(definite_value - integral) > eps:
        step = (b - a) / iterations
        integral = 0
        for i in range(iterations):
            integral += step * f(a + (i - 1) * step)
        #for plotting we choose first 100 iterations for better look of the plot
        if iterations < 100:
            errors.append(definite_value - integral)
            niter.append(iterations)
        iterations *= 2
    print("definite hand-calculated value : {} -- {} : value of the "
          "midpoint integration".format(definite_value, integral))
    return niter, errors

def midpoint_rule_for_sin(f, a, b, eps):
    definite_value = abs(integrate.quad(f, a, b)[0])
    iterations = 2
    integral = 0
    while abs(definite_value - integral) > eps:
        step = (b - a) / iterations
        integral = 0
        for i in range(iterations):
            if a + (i - 1) * step != 0:
                integral += step * f(a + (i - 1) * step)
            else:
                integral += step * 1
        iterations *= 2
    print("definite hand-calculated value : {} -- {} : value of the "
          "midpoint integration".format(definite_value, integral))

    return iterations



def main():

    print('Working with x**3 function. '
          'Evaluation of the accuracy of one-sided differentiation '
          'over the list of different values of "h" at the point 0.')
    x0 = 0
    for h in [1e-2, 1e-3, 1e-4, 1e-5]:
        err = abs(one_sided_deriv(f_test_1(x=symbols('x')), x0, h) - 0)# 0 is the analytically
        # predicted value of derivative at point 0.
        print("%5f -- %7.4g" % (h, err))
    print('We definetely see that the error is decreasing as O(h^(2)). '
          'Changing h as h/10 makes changing of the error as err/100')

    print('We see that for h -> 0, der(f) -> 0, and 0 is the analytically predicted value')


    print('Working with x**2*log(x) function. '
          'Evaluation of the accuracy of two-point differentiation '
          'over the list of different values of "h" at the point 1')
    x0 = 1
    for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        err = abs(two_point_deriv(f_test_2(x=symbols('x')), x0, h) - der_f_test_2(x0))
        print("%5f -- %7.4g" % (h, err))
    print('We again definetely see that the error is decreasing as O(h^(2)). '
          'Changing h as h/10 makes changing of the error as err/100')

    print('Working with x**2*log(x) function. '
          'Evaluation of the accuracy of three-point differentiation over the list of'
          'different values of "h" at the point 1')
    for h in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        err = abs(three_point_deriv(f_test_2(x=symbols('x')), x0, h) - der_f_test_2(x0))
        print("%5f -- %7.4g" % (h, err))
    print('We again definetely see that the error is decreasing as O(h^(2)). '
          'Changing h as h/10 makes changing of the error as err/100')


    print('We see that approximate critical h for two_point method is h = 1e-7, and for three_point method is h = 1e-6. '
          'After reaching the critical h, the accuracy stops increasing and starts decreasing, and so, '
          'visaversa, the error starts increasing for h less than the critical.')



    print('Working with x**2*log(x) function. Evaluation of the accuracy of '
          'three-point differentiation over the list of different values of "h" at the point 0,'
          ' assuming the value at extreme point is a pre-calculated limit')
    x0 = 0
    for h in [1e-2, 1e-3, 1e-4, 1e-5]:
        err = abs(three_point_deriv(f_test_2(x=symbols('x')), x0, h) - der_f_test_2(x0))
        print("%5f -- %7.4g" % (h, err))
    print('Error now is decreasing as O(h). Changing h as h/10 makes changing of the error as err/10')



    print('Now lets integrate a function f = x**2 from 0 to 1 using midpoint rule with the error of 0.0001')

    # for plotting we choose first 100 iterations for better look of the plot
    niter, errors = midpoint_rule(lambda x: x**2, 0, 1, 0.0001)

    print('Now lets integrate f = sin(sqrt(x)) / x from 0 to 1 with the error of 0.0001, previously transformed it '
          'into g = 2 * sin(x) / x and created a special midpoint_rule function knowing that the limit of sin(x)/x '
          'at x = 0 is 1. '
          'It is equal to meshing the grid at x in infinitesimal neighbourhood of 0.')
    number_of_iterations = midpoint_rule_for_sin(lambda x: 2*(sin(x) / x), 0, 1, 0.0001)
    print('Number of iterations for achieving the accuracy of 0.0001 for this function is {}, '
          'which is equal to 2^(n), where n = {}'.format(number_of_iterations, log(number_of_iterations, 2)))

    plt.figure()
    plt.title('Errors evaluation. Looks like O(n^(-2))')
    plt.xlabel('Number of iterations)')
    plt.ylabel('Error')
    plt.grid()
    plt.plot(niter, errors, label='error plot')
    plt.plot(niter, [iter ** (-2) for iter in niter], 'r', label='iter ^ (-2)')
    plt.legend()
    plt.show()



if __name__ == main():
    main()









