import sys,os
sys.path.append(os.pardir)
import numpy as np
from Chapter4.twolayernet import Twolayernet
from dataset.mnist import load_mnist
import matplotlib.pylab as plt

#full
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#train 10000 test 1000
x_train = x_train[:5000]
t_train = t_train[:5000]
x_test = x_test[:500]
t_test = t_test[:500]

train_loss_list = []
train_acc_list = []
test_acc_list = []

#hyper parameter
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 30
learning_late = 0.5

iter_per_epoch = max(train_size / batch_size, 1)
network = Twolayernet(input_size=784, hidden_size=50, output_size=10)

#start_i = 0
#start_loss = 100
#line, = plt.plot(start_i,start_loss)
#plt.ion()

for i in range(iters_num):
    #mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #calc grad
    grad = network.learn(x_batch, t_batch)

    #fix weight
    alpha = (np.exp(-i/iters_num) * learning_late)
    print(alpha)
    for key in ('W1','W2','b1','b2'):
        network.params[key] -= alpha * grad[key]

    #log
    loss = network.loss(x_batch, t_batch)
    print(i,":loss:",loss)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train accuracy:", train_acc)
        print("test accuracy:", test_acc)


    #a, = plt.plot([start_i,i+1],[start_loss, train_loss_list[i]], color="blue")

    #start_i = i + 1
    #start_loss = train_loss_list[i]
    #plt.ylim(0,10)
    #plt.xlim(0, ((i//1000) * 1000) + 1000)
    #plt.draw()
    #plt.pause(0.01)