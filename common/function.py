import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# x: batch
def cross_entropy_error_onehot(x, one_hot_label):
    if x.ndim == 1:
        one_hot_label = one_hot_label.reshape(1, one_hot_label.size)
        x = x.reshape(1, x.size)


    batch_size = x.shape[0]
    return -np.sum(one_hot_label * np.log(x)) / batch_size


def main():
    _test_softmax()


def _test_softmax():
    print("softmax()")
    x = np.array([1, 2, 3])
    softmax_out = softmax(x)
    print(softmax_out)


if __name__ == "__main__":
    main()
