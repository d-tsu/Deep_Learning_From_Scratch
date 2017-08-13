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
print("A * 10 =\n", A * 10)

# numpy broadcast
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print("A * B = \n", A * B)

# Access array
X = np.array([[51, 55],[14,19],[0,4]])
print("X = \n", X)
print("X[0] = ", X[0])
print("X[0][1] = ", X[0][1])

# for
print("for i in X: print(i)")
for i in X:
    print(i)
X = X.flatten()
print(X)
print(X[np.array([0,2,4])])
