import numpy as np


def function_2(x):
    return np.sum(x**2)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        print(tmp_val)
        # f(x+h) の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        print(fxh1)
        # f(x-h) の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        print(fxh2)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # もとのx[idx]に戻す

    return grad
