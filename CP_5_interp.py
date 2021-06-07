import numpy as np
import matplotlib.pyplot as plt



#интерполятор Лагранжа для выборки (X_test,y) и t как точки из области интерполирования
def lagranz(X_test, y, t):
    P = 0
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        m = len(X_test)
        for i in range(m):
            if i == j:
                pass
            else:
                p1 *= (t - X_test[i])
                p2 *= (X_test[j] - X_test[i])
        P += y[j] * p1 / p2
    return P

#равномерная сетка для m точек, по которым ведется интерполяция, и границ области интерполирования a и b
def uniform_grid(m, a, b):
    X0 = a
    X_test_all = [X0]
    while len(X_test_all) <= m:
        X0 += (b-a)/(m-1)
        X_test_all.append(X0)
    return X_test_all

#Лагранжева интерполяция по равномерной сетке
def Lagrange_interpolation():
    X_test_all = uniform_grid(5, np.pi/2, np.pi)
    X_test = []
    i = 0
    plt.figure()
    plt.title('Lagrange uniform grid interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    Y = []
    X = []
    for x in np.arange(np.pi / 2, np.pi, 0.1):
        X.append(x)
        Y.append(x ** 2 * np.cos(x))
    plt.plot(X, Y, 'g', linewidth=2, linestyle='-')
    for m in (1, 2, 3, 4, 5):
        m = m
        while True:
            if len(X_test) == m:
                break
            X_test.append(X_test_all[i])
            i += 1

        Y_test = [x**2 * np.cos(x) for x in X_test]
        plt.scatter(X_test, Y_test, color='g')
        X_under_interpolation = np.linspace(np.pi/2, np.pi, 100)
        Y_lagranz = [lagranz(X_test, Y_test, i) for i in X_under_interpolation]
        plt.plot(X_under_interpolation, Y_lagranz, color='r', linestyle='--')
        if m == 3:
            global data_L
            data_L = X_test[:3], Y_test, X_under_interpolation, Y_lagranz
    plt.show()

#создание сетки нулей полиномов Чебышёва для n точек и границ зоны left и right (с графическим представлением)
def Chebyschev_nodes(n, left, right):
    X_test_CHEBYSCHEV = []
    m = 1
    while m <= n:
        node = (left + right)/2 + (right - left)/2 * np.cos((2*m - 1)/(2*n)*np.pi)
        X_test_CHEBYSCHEV.append(node)
        m += 1
    X_test_CHEBYSCHEV.reverse()
    return X_test_CHEBYSCHEV

#интерполяция методом Лагранжа по сетке нулей полиномов Чебышёва (с графическим представлением)
def Chebyschev_interpolation():
    Chebyschev_test = Chebyschev_nodes(5, np.pi/2, np.pi)
    X_test = []
    i = 0
    plt.figure()
    plt.title('Chebyshev node grid interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    Y = []
    X = []
    for x in np.arange(np.pi / 2, np.pi, 0.1):
        X.append(x)
        Y.append(x ** 2 * np.cos(x))
    plt.plot(X, Y, 'g', linewidth=2, linestyle='-')
    for m in (1, 2, 3, 4, 5):
        m = m
        while True:
            if len(X_test) == m:
                break
            X_test.append(Chebyschev_test[i])
            i += 1
        Y_test = [x**2 * np.cos(x) for x in X_test]
        plt.scatter(X_test, Y_test, color='g')
        X_under_interpolation = np.linspace(np.pi/2, np.pi, 100)
        Y_lagranz = [lagranz(X_test, Y_test, i) for i in X_under_interpolation]
        plt.plot(X_under_interpolation, Y_lagranz, color='r', linestyle='--')
        if m == 3:
            global data_Ch
            data_Ch = X_test[:3], Y_test, X_under_interpolation, Y_lagranz
    plt.show()

#сравнение качества интерполяции для точек Чебышёвской сетки и равномерной сетки для 3 точек
def comparison_at_m_3():
    plt.figure()
    Y = []
    X = []
    for x in np.arange(np.pi/2, np.pi, 0.1):
        X.append(x)
        Y.append(x ** 2 * np.cos(x))
    plt.plot(X, Y, 'g', linewidth=2, linestyle='-')
    plt.title('Comparison at m = {} for uniform mesh and Chebyshev nodes'.format(3))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(data_L[0], data_L[1], linewidths=3)
    plt.scatter(data_Ch[0], data_Ch[1], linewidths=0.5)
    plt.plot(data_L[2], data_L[3], linestyle='--')
    plt.plot(data_Ch[2], data_Ch[3], linestyle='--')
    plt.grid()
    plt.show()

#запуск программы
def main():
    Lagrange_interpolation()
    Chebyschev_interpolation()
    comparison_at_m_3()

if __name__ == main():
    main()

#примечание: ввиду особенностей интерполируемой области и интерполируемой функции разница между интерполяцией
#по равномерной метрике и по метрике полиномов Чебышёва получилась почти незаметная.