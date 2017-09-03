import numpy as np


def numerical_diff_bad(f, x):
    h = 1e-4
    return (f(x + h) - f(x)) / h


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def func(x):
    return x ** 3 - x ** 2


def main():
    print("numerical diff bad example")
    print(numerical_diff_bad(func, 10))
    print("numerical diff good expample")
    print(numerical_diff(func, 10))


if __name__ == "__main__":
    main()
