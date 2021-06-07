import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import qr

#Сначала напишем свою функцию для вычисления нормы вектора (не захотелось использовать np.linalg.norm)

def vec_norm(dim , vec):
    result = 0
    for i in range(dim):
       result = result + vec[i] * vec[i]
    vector_norm = np.sqrt(result)
    return vector_norm

#функция преобразования Хаусхолдера для матрицы
def householder(a):
    v = a / (a[0] + np.copysign(vec_norm(1,a), a[0]))
    v[0] = 1
    h = np.eye(a.shape[0]) - (2 / (v @ v)) * (v[:, None] @ v[None, :]) #формула преобразования Хаусхолдера h = I - 2vv
    return h

#собственно, преобразование QR
def qr_decomp(a):
    a1 = np.array(a, copy=True, dtype=float) #считываем матрицу
    m, n = a1.shape  #измерения матрицы
    q = 0 #вводим для красоты кода q и r до цикла(необязательное действие)
    r = 0
    for i in range(n - (m == n)):
        h = np.eye(m)  #создаём единичную матрицу
        h[i:, i:] = householder(a1[i:, i]) #выполняем хаусхолдеровское преобразование для нашей матрицы, обрезанной после
        q = np.eye(m) @ h  #q - это Хаусхолдерова матрица, умноженная на единичную (в нужной размерности)
        r = h @ a1  # r - это Хаусхолдерова матрица, умноженная на исходную
    return q, r





#проверка кода


np.set_printoptions(suppress=True)


# Test II.1 (20% of the total grade)


rndm = np.random.RandomState(1234)
a = rndm.uniform(size=(5, 3))
q, r = qr_decomp(a)


assert_allclose(np.dot(q, q.T), np.eye(5), atol=1e-10)


# test the decomposition itself
assert_allclose(np.dot(q, r), a)


#проверим сходство с библиотечным qr
qq, rr = qr(a)

assert_allclose(np.dot(qq, rr), a)

