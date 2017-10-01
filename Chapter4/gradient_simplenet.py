import numpy as np
import sys, os

from pandas.util.testing import network

sys.path.append(os.pardir)
from common.function import *
from common.gradient import *


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)


    def predict(self, x):
        return np.dot(x, self.W)


    def loss(self,x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error_onehot(y, t)
        return loss



if __name__ == "__main__":
    net = simpleNet()
    print(net.W)
    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))
    t = np.array([0,0,1])
    f = lambda W: net.loss(x, t)
    print("W id:", id(net.W))
    dw = numerical_gradient(f, net.W)
