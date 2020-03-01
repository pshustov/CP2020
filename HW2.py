import numpy as np
from numpy.testing import assert_allclose
np.set_printoptions(suppress=True)

def householder(vec):
    vec = np.asarray(vec, dtype=float)
    if vec.ndim != 1:
        raise ValueError("vec.ndim = %s, expected 1" % vec.ndim)
    N = vec.shape[0]
    y = np.zeros(N)
    y[0] = np.linalg.norm(vec)
    u = (vec - y) / np.linalg.norm(vec - y)
    H = np.eye(N) - 2 * np.outer(u, u)
    return y, H


# Test I.1 (10% of the total grade).
v = np.array([1, 2, 3])
v1, h = householder(v)
assert_allclose(np.dot(h, v1), v)
assert_allclose(np.dot(h, v), v1, atol=1e-15)

# Test I.2 (10% of the total grade).
rndm = np.random.RandomState(1234)
vec = rndm.uniform(size=7)
v1, h = householder(vec)
assert_allclose(np.dot(h, v1), vec)


def qr_decomp(a):
    a1 = np.array(a, copy=True, dtype=float)
    m, n = a1.shape
    Q1 = np.eye(m)
    for i in range(min(m - 1, n)):
        Q = np.eye(m)
        vec = a1[i:, i]
        T = householder(vec)[1]
        Q[i:, i:] = T
        Q1 = Q1 @ Q.T
        a1 = Q @ a1
    return Q1, a1

rndm = np.random.RandomState(1234)
a = rndm.uniform(size=(5, 3))
q, r = qr_decomp(a)

# test that Q is indeed orthogonal
assert_allclose(np.dot(q, q.T), np.eye(5), atol=1e-10)

# test the decomposition itself
assert_allclose(np.dot(q, r), a)


from scipy.linalg import qr
qq, rr = qr(a)
assert_allclose(np.dot(qq, rr), a)
assert_allclose(np.dot(qq, rr), qr_decomp(a)[0] @ qr_decomp(a)[1])

# print (qq, "\n")
# print(qr_decomp(a)[0])

# print (rr, "\n")
# print(qr_decomp(a)[1])
# Видно, что qq и q, а также rr и r не совпадают, однако их произведения возвращают одну матрицу
# Скорее всего так происходит из-за другого метода разложения


def qr_decomp_esp(a):
    a1 = np.array(a, copy=True, dtype=float)
    m, n = a1.shape
    vec1 = []
    for i in range(n):
        vec = a1[i:, i]
        N = vec.shape[0]
        y = np.zeros(N)
        y[0] = np.linalg.norm(vec)
        u = (vec - y) / np.linalg.norm(vec - y)
        vec1.append(u)
        a1[i:, i:] -= 2 * np.outer(u, u @ a1[i:, i:])
    return a1, vec1


def qt_rec(vec1, a):
    m, n = a.shape
    QT = np.eye(m)
    for i in range(n):
        QT[n-i-1:, n-i-1:] -= 2 * np.outer(vec1[n-i-1], vec1[n-i-1] @ QT[n-i-1:, n-i-1:])
    return QT
a1, vec1 = qr_decomp_esp(a)
qt = qt_rec(vec1, a)

assert_allclose(qt.T @ qt, np.eye(5), atol=1e-10)
assert_allclose(qt.T @ a, a1, atol=1e-10)