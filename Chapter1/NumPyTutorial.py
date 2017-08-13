import numpy as np

x = np.array([1, 2, 3])
print(x)
print(type(x))

y = np.array([2, 4, 6])
print("x + y = ", x + y)
print("x - y = ", x - y)
print("x * y = ", x * y)
print("x / y = ", x / y)

print("x / 2 = ", x / 2.0)

# matrix
A = np.array([[1, 2], [3, 4]])
print(A)

B = np.array([[3, 0],[0,6]])
print("A + B =\n", A + B)
print("A * B =\n", A * B)

