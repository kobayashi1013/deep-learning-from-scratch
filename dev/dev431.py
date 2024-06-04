import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def function_2(a, x):
    return a * x

plt.xlabel("X")
plt.ylabel("f(x)")

x = np.arange(0.0, 20.0, 0.1)
y1 = function_1(x)
y2 = function_2(numerical_diff(function_1, 5), x)
y3 = function_2(numerical_diff(function_1, 10), x)

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)

plt.show()

#print(numerical_diff(function_1, 5))
#print(numerical_diff(function_1, 10))