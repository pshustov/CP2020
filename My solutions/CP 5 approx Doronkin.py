import numpy as np
from numpy.linalg import inv, solve, qr
import matplotlib.pyplot as plt

#считываем данные
x = [-1, -0.7, -0.43, -0.14, -0.14, 0.43, 0.71, 1, 1.29, 1.57, 1.86, 2.14, 2.43, 2.71, 3]
y = [-2.25, -0.77, 0.21, 0.44, 0.64, 0.03, -0.22, -0.84, -1.2, -1.03, -0.37, 0.61, 2.67, 5.04, 8.90]
y = np.array(y)
n = len(x)


#функция создания design матрицы m*n для набора x длины n и максимальной стпенью полинома, равной m
def design_matrix(x, m):
    n = len(x)
    a = np.arange(m)
    design_matrix = np.zeros(m)
    for i in range(n):
        line = np.full(m, x[i])
        line_of_design_matrix = np.power(line, a)
        design_matrix = np.vstack((design_matrix, line_of_design_matrix))
    design_matrix = np.delete(design_matrix, 0, axis=0)
    return design_matrix


#функция решения нормального уравнения для поступающей матрицы А (будет в дальнейшем поступать design матрица)
#Нормальное уравнение: (A.T*A)*beta_vector = A.T*y
#=> beta_vector = (A.T*A)^(-1)*A.T*y
def normal_equasion_solution(A, y):
    beta_vector = inv(A.transpose().dot(A)).dot(A.transpose()).dot(y) #решаем уравнение относительно бета_вектора
    return beta_vector


#функция поиска оптимального значения для m для определенной выборки (x, y)
def optimal_value_of_m(m, y, x):
    beta_vector = normal_equasion_solution(design_matrix(x, m), y)
    y_approx = []

    for j in range(len(x)):
        beta = beta_vector[0] * (x[j] ** 0)
        if len(beta_vector) > 1:
            for i in range(1, len(beta_vector)):
                beta += beta_vector[i]*(x[j]**i)
        y_approx.append(beta)

    D = 0
    for k in range(n):
        D += (y_approx[k] - y[k])**2
    D = 1 / (n - m) * D
    return D


#Найдём через QR (альтернатива решению нормального уравнения)
def QR_factorization_for_Lineal_least_squares(x, y, m):
    n = len(x)
    A = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            A[i][j] = x[i] ** j
    Q, R = qr(A, mode='complete')
    R1 = R[:m]
    f = np.dot(Q.T, y)[:m]
    beta_vector_with_QR = solve(R1, f)
    return beta_vector_with_QR

#Cравнение значений бета-вектора, найденного через решение нормального уравнения и через QR
#Спойлер: как будет видно потом в выводе программы, различие практически нулевое и уже связано скорее всего
#в большей степени с машинной точностью
def comparison_QR_normal_equasion(x, y, m):
    QR = QR_factorization_for_Lineal_least_squares(x, y, m)
    NormalEquasion = normal_equasion_solution(design_matrix(x, m), y)
    print('vector of squared_difference between beta_vectors from QR and NE = {}. '
          'So we see that this differrence is practically equal to zero for every '
          'element of beta_vector'.format((QR - NormalEquasion)**2))

#Функция, которая нарисует график зависимости дисперсии от m и выдаст итоговое оптимальное значение m. Тут я уже учел,
# что значения m, близкие к n, будут давать очень большую дисперсию и поэтому я ограничился максимальным m = 12
# для нашего набора 15 точек (чтобы график был красивее и информативнее)
def final_m():
    D = []
    mm = []
    for m in range(1, 13):
        D.append(optimal_value_of_m(m, y, x))
        mm.append(m)
    plt.figure()
    plt.title('The dispersion for polinoms with different m (1 < m < 12)')
    plt.plot(mm, D)
    plt.xlabel('m')
    plt.ylabel('Dispersion')
    print('Best fitting polinom has m = {} and the dispersion = {}'.format(np.argmin(D)+1, np.min(D)))

    final_m = np.argmin(D)+1

    return final_m










#функция вывода массива итоговых значений полиномиальной аппроксимации для выборки (x, y) с определенным m
def final_y_approx(x, y, m):
    beta_vector = normal_equasion_solution(design_matrix(x, m), y)
    y_approx = []
    for j in range(len(x)):
        beta = beta_vector[0] * (x[j] ** 0)
        for i in range(1, len(beta_vector)):
            beta += beta_vector[i]*(x[j]**i)
        y_approx.append(beta)
    return y_approx








#основная программа, выводящая в итоге пользователю два графика (дисперсия от m для нашей выборки и всех полиномов и исходный набор точек
#Четко видно, что наша оптимальная степень полинома, равная 12, очень слабо отличается от более ранних аппроксимаций
#(хватает для достаточной точности полинома степени 4)
def main():
    comparison_QR_normal_equasion(x, y, 5)  # возьмём m = 5
    y_approx = final_y_approx(x, y, final_m())
    plt.figure()
    plt.title('Different fitting polinoms and experiment points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y, label='experiment points')
    plt.plot(x, y_approx, color = 'r', linewidth = 2, label='best polinom')
    for m in range(1, 15):
        y_approx = final_y_approx(x, y, m)
        plt.plot(x, y_approx, color='g', linewidth = 0.5, linestyle = '--')
    plt.legend(loc='best')
    plt.show()


#запуск программы
if __name__ == main():
    main()











