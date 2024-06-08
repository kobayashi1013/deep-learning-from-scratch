import numpy as np

def softmax(x):
    #オーバーフロー対策
    c = np.max(x, axis=1, keepdims=True)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)
    return exp_a / sum_exp_a

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
        
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size