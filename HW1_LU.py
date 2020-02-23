import numpy as np

np.set_printoptions(precision=3)
N = 6
a = np.zeros((N, N), dtype=float)
for i in range(N):
    for j in range(N):
        a[i, j] = 3. / (0.6*i*j + 1)
a1 = np.copy(a)
a1[1,1] = 3
# print(np.linalg.matrix_rank(a))


def diy_lu(a):
    N = a.shape[0]
    u = a.copy()
    L = np.eye(N)
    for j in range(N - 1):
        lam = np.eye(N)
        gamma = u[j + 1:, j] / u[j, j]
        lam[j + 1:, j] = -gamma
        u = lam @ u
        lam[j + 1:, j] = gamma
        L = L @ lam
    return L, u


def test_minor(a):
    M = np.zeros(a.shape)
    for i in range(N):
        M[i,i] = np.linalg.det(a[:i, :i])
        if M[i, i] ==0:
            return M, False
    return M, True


print ("a:", test_minor(a), "\n")
print ("a1:", test_minor(a1), "\n")  # Видно, что у данной матрицы отсутсвует главный минор 3-го порядка


def pivoting(a):
    N = a.shape[0]
    R = np.eye(N)
    M = a.copy()
    for i in range(N):
        E = np.eye(N)
        S = np.argmax(np.abs(M[:, i][i:])) + i
        E[[i, S]] = E[[S, i]]
        R = E @ R
        M = E @ M
    return R

def lup(a):
    N = a.shape[0]
    L = np.eye(N)
    p = np.eye(N)
    u = a.copy()
    for j in range(N - 1):
        lam = np.eye(N)
        gamma = u[j + 1:, j] / u[j, j]
        lam[j + 1:, j] = -gamma
        u = lam @ u
        M = pivoting(u).T
        u = u @ M
        lam[j + 1:, j] = gamma
        L = L @ lam
        p = p @ M
    u.round()
    return L, u, p.T

def back(L, U, P):
    return L @ U @ P

L,U,P = lup(a)
L1,U1,P1 = lup(a1)
print("L:", L,"\n","U:", U,"\n","P:", P,"\n")
print("L1:", L1,"\n","U1:", U1,"\n","P1:", P1,"\n")
print(back(L,U,P),"\n")
print(back(L1,U1,P1),"\n")
np.allclose(back(L,U,P),back(L1,U1,P1))