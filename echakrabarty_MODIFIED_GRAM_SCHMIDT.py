# Esha Chakrabarty
# 920884665

import numpy as np

matrix2 = np.array([[1, 4, 1, 0],
                    [0, 1, 0, 2],
                    [0, 0, 2, 0],
                    [1, 0, 1, 1]])


def mgs(matrix):
    
    m, n = matrix.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = matrix[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i].T, v)
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q


print(mgs(matrix2))


