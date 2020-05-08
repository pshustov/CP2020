import numpy as np
import matplotlib.pyplot as plt

def euler_solve(lam, u0, T, dt):
    """Решает $du/dt = \lambda u$ на $0 < t < T$ с $u(t=0) = u0$ при помощи явного метода Эйлера."""
    num_steps = int(T/dt)
    tt = np.arange(num_steps+1)*dt
    y = np.empty(num_steps+1)
    y[0] = u0
    for k in range(num_steps):
        y[k + 1] = y[k] + dt*lam*y[k]
    return tt, y



def implicit_euler_solve(lam, u0, T, dt):
    num_steps = int(T/dt)
    tt = np.arange(num_steps+1)*dt
    y = np.empty(num_steps+1)
    y[0] = u0
    #общий вид преобразованной неявной схемы для ОДУ, линейного относительно y: (y[k+1] - y[k]) / dt  =  lam * y[k+1]
    #будем писать сразу преобразованное выражение (выразим y[k+1])
    for k in range(num_steps):
        y[k + 1] = y[k] / (1 - dt * lam)
    return tt, y


def Test_0_0():
    for i in (0.01, 0.1, 1):
        lam = - 0.5
        tt, y = euler_solve(lam, u0=1.0, T=5, dt=i)
        tt_I, y_I = implicit_euler_solve(lam, u0=1.0, T=5, dt=i)
        plt.figure()
        plt.plot(tt, y, 'o--', label='numeric solution')
        plt.plot(tt, np.exp(lam*tt), '-', lw=2, label='ground truth')
        plt.plot(tt_I, y_I, 'o--', label='implicit_numeric solution')

        plt.legend(loc='best')
        plt.grid(True)



def matrix_euler_solve(A, u0, T, dt):
    num_steps = int(T / dt)
    tt = np.arange(num_steps + 1) * dt
    y = np.empty([num_steps + 1, 2])
    y[0] = u0
    for k in range(num_steps):
        y[k + 1] = y[k] + dt * np.dot(A, y[k])
    return tt, y

def Test_2():

    A = np.array([[-10, 10], [32, -499]])
    u0 = np.array([1, 0])
    tt, y = matrix_euler_solve(A, u0=u0, T=5, dt=0.01)

    eigvals_array = np.linalg.eigvals(A)
    np.testing.assert_array_less(eigvals_array, 0)
    S = min(np.real(eigvals_array)) / max(np.real(eigvals_array))
    print(S)
    if 10 < S < 100:
        print('Система средней жесткости (обычно система считается жесткой при S от 100 и выше)')
    elif S > 100:
        print('Система жесткая')
    else:
        print('Система нежесткая')

    tt_imp, y_imp = matrix_implicit_euler_solve(A, u0=u0, T=5, dt=0.01)


def matrix_implicit_euler_solve(A, u0, T, dt):

    num_steps = int(T/dt)
    tt = np.arange(num_steps + 1) * dt
    y = np.empty((num_steps + 1, 2))
    y[0] = u0
    y_1 = np.empty((num_steps + 1, 2))
    y_1[0] = u0
    for k in range(num_steps):
        #можно решить двумя методами, через np.linalg.inv напрямую, или через np.linalg.solve
        y[k+1] = np.dot(y[k], np.linalg.inv(np.eye(2) - dt * A))
        #а можно через np.linalg.solve
        y_1[k+1] = np.linalg.solve((np.eye(2) - dt * A), y[k])
        #второй метод будет точнее, так как операция solve содержит в себе меньше операций округления,
        #а значит большую точность. Это так, так как solve вообще не вычисляет обратную матрицу,
        #а использует внутри себя LU-разложение. Поэтому остановимся на втором.

    return tt, y_1






def Runge_Kutta_2nd_order(A, u0, T, dt):
    num_steps = int(T/dt)
    tt = np.arange(num_steps + 1) * dt
    y = np.empty([num_steps + 1, 2])
    y[0] = u0
    for k in range(num_steps):
        y[k+1] = y[k] + dt * np.dot(A, (y[k] + dt * np.dot(A, y[k]) / 2))
    return tt, y


def Test_3():

    w = 1

    A = np.array([[0, - w ** 2], [1, 0]])
    u0 = np.array([0, 1])

    plt.figure(figsize=[5, 5])
    plt.title('Euler E(t)')
    for dt in (0.1, 0.001):
        tt, y = matrix_euler_solve(A, u0, 5, dt)
        E = y[:, 1] ** 2 / 2 + w ** 2 * y[:, 0] ** 2 / 2
        plt.plot(tt, E, label="step = {}".format(dt))
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize=[5, 5])
    plt.title('Runge-Kutta 2-nd order')
    for dt in (0.1, 0.001):
        tt_RK, y_RK = Runge_Kutta_2nd_order(A, u0, 5, dt)
        E_RK = y_RK[:, 1] ** 2 / 2 + w ** 2 * y_RK[:, 0] ** 2 / 2
        plt.plot(tt_RK, E_RK, label="step= {}".format(dt))
    plt.grid()
    plt.legend()
    plt.show()


def main():
    Test_0_0()
    Test_2()
    Test_3()

if __name__ == main():
    main()