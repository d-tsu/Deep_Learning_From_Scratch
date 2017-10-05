import sys,os
sys.path.append(os.pardir)
import numpy as np
from Chapter4.twolayernet import Twolayernet
from dataset.mnist import load_mnist
import matplotlib.pylab as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

#hyper parameter
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 20
learning_late = 0.2

network = Twolayernet(input_size=784, hidden_size=20, output_size=10)

start_i = 0
start_loss = 100

#line, = plt.plot(start_i,start_loss)
plt.ion()

for i in range(iters_num):
    #mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #calc grad
    grad = network.learn(x_batch, t_batch)

    #fix weight
    for key in ('W1','W2','b1','b2'):
        network.params[key] -= (np.exp(-i/iters_num) * learning_late) * grad[key]

    #log
    loss = network.loss(x_batch, t_batch)
    print(i,":loss:",loss)
    train_loss_list.append(loss)
    #print(i)
    #print(train_loss_list)
    a, = plt.plot([start_i,i+1],[start_loss, train_loss_list[i]], color="blue")
    #print(id(a))
    start_i = i + 1
    start_loss = train_loss_list[i]
    plt.ylim(0,10)
    plt.xlim(0, ((i//1000) * 1000) + 1000)
    #plt.draw()
    plt.pause(0.01)







