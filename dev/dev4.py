import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

class Utils:

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(a):
        max = np.max(a)
        exp_a = np.exp(a - max)
        sum_exp_a = np.sum(exp_a)
        return exp_a / sum_exp_a
    
    def cross_entropy_error(y, t):
        
        if y.ndim == 1:
            y = y.reshape(1, y.size)
            t = t.reshape(1, t.size)
        
        #OneHot無効化
        if y.size == t.size:
            t = t.argmax(axis = 1)
        
        batch_size = y.shape[0]
        
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    
    def numerical_gradient(f, x):

        h = 1e-4
        grad = np.zeros_like(x)

        it = np.nditer(x, flags = ['multi_index'])
        while not it.finished:
            
            idx = it.multi_index
            tmp_val = x[idx]

            x[idx] = tmp_val + h
            fxh1 = f(x)

            x[idx] = tmp_val - h
            fxh2 = f(x)

            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val
            it.iternext()
        
        return grad
    
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def numerical_gradient(self, x, t):

        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = Utils.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = Utils.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = Utils.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = Utils.numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def loss(self, x, t):

        y = self.predict(x)

        return Utils.cross_entropy_error(y, t)


    def predict(self, x):

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = Utils.sigmoid(a1)
        
        a2 = np.dot(z1, W2) + b2
        y = Utils.softmax(a2)

        return y

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

train_loss_list = []

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
lr = 0.1

input_size = x_train.shape[1]
hidden_size = 1
output_size = t_train.shape[1]

network = TwoLayerNet(input_size, hidden_size, output_size)

for i in range(iters_num):

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= lr * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    if i % 10 == 0:
        print(i / 10)

x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
plt.show()