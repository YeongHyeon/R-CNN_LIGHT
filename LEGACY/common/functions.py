# coding: utf-8
import numpy as np
from scipy.special import expit, logit

def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    #return 1 / (1 + np.expit(-x))
    return expit(x) # expit (x) = 1 / (1 + exp (-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    #grad = np.zeros(x) # origin
    grad = np.zeros(x.shape) # modify
    grad[x>=0] = 1
    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # Prevent overflow
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    # y[np.arange(batch_size), t] -> [0, t[0]], [1, t[1]], [2, t[2]], [3, t[3]]...
    """
    numpy.log(x) means ln(x)
    ln(x) = log(x) * log(e)
    e = lim(1+x)^(1/x)  -(when x->0)
    """
    # np.log(y) -> np.log([n, m]) -> [np.log(n), np.log(m)]
    # np.sum(log) -> np.log(n) + np.log(m) + ...
    delta = 1e-7 #prevent infinite value
    nn_out = y[np.arange(batch_size), t]
    return -np.sum(np.log(nn_out+delta)) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
