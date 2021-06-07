import numpy as np
from numpy.testing import assert_allclose
#Сначала напишем свою функцию для вычисления нормы вектора (не захотелось использовать np.linalg.norm)

def vec_norm(dim , vec):

    result = 0
    for i in range(dim):
       result = result + vec[i] * vec[i]
    vector_norm = np.sqrt(result)
    return vector_norm

def householder(vec):

    vec = np.asarray(vec, dtype=float)  #считываем вектор, к которому хотим применить преобразование

    if vec.ndim != 1:  #исключаем многомерные вектора
        raise ValueError("vec.ndim = %s, expected 1" % vec.ndim)

    n = len(vec)  #оно же vec.shape[0] - длина вектора (векторы принимаем только одномерные)
    I = np.eye(n) #единичная матрица нужной размерности
    Z = np.zeros(n) #матрица с зануленными элементами нужной размерности
    Z[0] = np.linalg.norm(vec)
    v = (vec - Z) / vec_norm(n, vec - Z)

    h = I - 2 * np.outer(v, v)#формула преобразования Хаусхолдера h = I - 2vv

    return Z, h


#выполняем проверку

v = np.array([1, 2, 3])
v1, h = householder(v)
assert_allclose(np.dot(h, v1), v, atol=1e-10) #так как assert allclose требует по умолчанию
                               #полного соответствия элементов, то введём здесь и в дальнейших тестах
                             #параметр atol, который позволяет assertallclose считать нулевой "почти нулевое" различие,
                                # так как логично что 10^-10 достаточно малое отклонение от нуля.
assert_allclose(np.dot(h, v), v1, atol=1e-10)

rndm = np.random.RandomState(1234)

vec = rndm.uniform(size=7)
v1, h = householder(vec)

assert_allclose(np.dot(h, v1), vec, atol=1e-10)


