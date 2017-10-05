import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.function import *
from common.gradient import *


class Twolayernet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        # forward
        layer_1 = np.dot(x, W1) + b1
        activation_1 = ReLU(layer_1)
        layer_2 = np.dot(activation_1, W2) + b2
        output = softmax(layer_2)
        return output

    def loss(self, x, t):
        output = self.predict(x)
        return cross_entropy_error_onehot(output, t)

    def accuracy(self, x, one_hot_label):
        output = self.predict(x)
        predict_label = np.argmax(output, axis=1)
        actual_label = np.argmax(one_hot_label, axis=1)
        accuracy = np.sum(predict_label == actual_label) / float(x.shape[0])
        return accuracy

    def learn(self, x, one_hot_label):
        loss_W = lambda W: self.loss(x, one_hot_label)
        grads = {}
        W = ["W1", "W2"]
        B = ["b1", "b2"]
        grads[W[0]] = numerical_gradient(loss_W, self.params[W[0]])
        grads[B[0]] = numerical_gradient(loss_W, self.params[B[0]])
        grads[W[1]] = numerical_gradient(loss_W, self.params[W[1]])
        grads[B[1]] = numerical_gradient(loss_W, self.params[B[1]])

        return grads


def main():
    net = Twolayernet(input_size=784, hidden_size=100, output_size=10)
    print(net.params["W1"].shape)
    print(net.params["W2"].shape)
    print(net.params["b1"].shape)
    print(net.params["b2"].shape)


if __name__ == "__main__":
    main()
