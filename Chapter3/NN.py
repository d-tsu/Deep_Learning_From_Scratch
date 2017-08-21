import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def creategraph(x, func):
    y = func(x)
    plt.plot(x, y)
    plt.show()

