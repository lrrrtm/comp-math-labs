import numpy as np
from scipy.linalg import lu_factor, lu_solve

N = 6
alpha = 4
gammas = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]


def compute_g(N):
    return np.array([2 ** (k - 4) for k in range(1, N + 1)])


def vector_norm(v):
    return np.sum(np.abs(v))


def build_Q_and_z(N, alpha, x, y, g):
    Q = np.zeros((N, N))
    z = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                Q[i, j] = alpha
            elif i < j:
                Q[i, j] = x
            else:
                Q[i, j] = y

        z[i] = (y * np.sum(g[:i]) if i > 0 else 0) + alpha * g[i] + (x * np.sum(g[i + 1:]) if i < N - 1 else 0)

    return Q, z


g = compute_g(N)
norm_g = vector_norm(g)

for gamma in gammas:
    x = 4 + gamma
    y = 4 - gamma
    Q, z = build_Q_and_z(N, alpha, x, y, g)

    print(f"\ngamma = {gamma}:")
    print(Q)

    lu, piv = lu_factor(Q)
    w = lu_solve((lu, piv), z)

    error = vector_norm(w - g) / norm_g
    print(f'||w-g||/||g|| = {error}')
