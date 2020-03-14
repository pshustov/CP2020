import numpy as np
import matplotlib.pyplot as plt

rndm = np.random.RandomState(1234)
n = 10
A = rndm.uniform(size=(n, n)) + np.diagflat([5]*n)
A1 = rndm.uniform(size=(n, n)) + np.diagflat([4]*n)
b = rndm.uniform(size=n)

def jacobi_iteration(A, b, n_iter):
    n = len(A)
    diag_1d = np.diag(A)
    B = -A.copy()
    np.fill_diagonal(B, 0)
    D = np.diag(diag_1d)
    invD = np.diag(1./diag_1d)
    BB = invD @ B
    c = invD @ b
    x0 = np.ones(n)
    x = x0
    # print(np.linalg.norm(BB))
    for _ in range(n_iter):
        x = BB @ x + c
    return x



xx = np.linalg.solve(A, b)

# Построим зависимость нормы ошибки от колличества итераций:
# for i in range(1, 30):
#     eps = np.linalg.norm(A @ jacobi_iteration(A1, b, i) - b)
#     plt.plot(i, eps, 'o')
# plt.show()
# Видно, что при добавке на диагонали >=5 (матрица A) зависимость экспоненциально убывающая,
# а при 4 (матрица A1) уже возрастающая из-за суммирующейся ошибки.


def seidel_iteration(A, b, n_iter):
    n = len(A)
    x = np.zeros(n)
    x_new = x
    for _ in range(n_iter):
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        x = x_new
    return x

k = 500
A2 = rndm.uniform(1, k, (n, n)) + np.diagflat([k]*n)
b2 = rndm.uniform(1, k, n)
# Если главную диагональ не увеличить относительно остальных, то метод будет также расходиться.
np.testing.assert_allclose(seidel_iteration(A2, b2, 100), np.linalg.solve(A2, b2))

# Строим ту же зависимость для метода Зейделя:
# for i in range(1, 30):
#     eps = np.linalg.norm(A2 @ seidel_iteration(A2, b, i) - b)
#     plt.plot(i, eps, 'o')
# plt.show()


def minimum_residual(A, b, n_iter=100):
    x = np.zeros(len(A))
    temp = np.empty(0)
    r_norm = np.empty(0)
    r_dev = np.empty(0)
    for _ in range(n_iter):
        r = A @ x - b
        r_norm = np.append(r_norm, np.linalg.norm(r))
        r_dev = np.append(r_dev, np.linalg.norm(x - np.linalg.solve(A, b)))
        tau = (r @ (A @ r)) / np.linalg.norm(A @ r) ** 2
        temp = np.append(temp, tau)
        x = x - tau * r
    return x, r_norm, r_dev, temp


x, r_norm, r_dev, t = minimum_residual(A2, b2)
np.testing.assert_allclose(x, np.linalg.solve(A2, b2))
# И та же зависимость для метода мнимальных остатков:
# for i in range(1, 30):
#     x1, r_norm1, r_dev1, t1 = minimum_residual(A2, b2, i)
#     eps = np.linalg.norm(A2 @ x1 - b2)
#     plt.plot(i, eps, 'o')
# plt.show()


range = np.arange(t.shape[0])
plt.plot(range, t, ".-")
plt.ylabel("тау")
plt.show()
plt.plot(range, r_norm)
plt.ylabel("норма отклонения")
plt.show()
plt.plot(range, r_dev)
plt.ylabel("Отклонение от истинного значения")
plt.show()
# Видно, что тау оссцилирует, но отклонение от истинного решения экспоненциально убывает от кол-ва итераций.