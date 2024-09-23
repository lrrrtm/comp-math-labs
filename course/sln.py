import numpy as np

A_matrix = np.array([[10, 1, 4, 0],
                     [1, 10, 5, -1],
                     [4, 5, 10, 7],
                     [0, -1, 7, 9]])

b_vector = np.array([5, 13, 29, 24])

solution = np.linalg.solve(A_matrix, b_vector)

A, B, F, G = solution
print(f"A = {A}\nB = {B}\nF = {F}\nG = {G}")
