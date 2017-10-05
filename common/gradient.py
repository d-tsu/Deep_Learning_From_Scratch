import numpy as np


def function_2(x):
    return np.sum(x ** 2)


def numerical_gradient(f, x):
    #print("numerical gradient x id:", id(x))
    h = 1e-4
    grad = np.zeros_like(x)
    #print("grad:", grad)
    #print(x.size)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 値を元に戻す
        it.iternext()
        # もとのx[idx]に戻す

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    print(x.dtype)
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
