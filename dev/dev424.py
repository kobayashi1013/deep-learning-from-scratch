import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

t = [1, 4]
y = [[0.1, 0.7, 0, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6]]

print(cross_entropy_error(np.array(y), np.array(t)))