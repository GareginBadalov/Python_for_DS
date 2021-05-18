import numpy as np


a = np.array([[1, 6], [2, 8], [3, 11],
              [3, 10], [1, 7]])
b = a.mean(axis=0)
a_centered = a - b
multiply = np.dot(a_centered[:, 0], a_centered[:, 1])
N = a.shape[0]

print(multiply / (N-1))
