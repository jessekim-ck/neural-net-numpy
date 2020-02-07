import numpy as np


class ReLU:

    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        A = np.fmax(0, Z)
        return A

    def backward(self, dA):
        dg = self.Z > 0
        dZ = dA*dg
        return dZ


class Sigmoid:

    def __init__(self):
        self.A = None

    def forward(self, Z):
        A = sigmoid(Z)
        self.A = A
        return A

    def backward(self, dA):
        dg = self.A*(1 - self.A)
        dZ = dA*dg
        return dZ


class TangentH:

    def __init__(self):
        self.A = None
    
    def forward(self, Z):
        A = tanh(Z)
        self.A = A
        return A
    
    def backward(self, dA):
        dg = 1 - self.A**2
        dZ = dA*dg
        return dZ


class Affine:

    def __init__(self, rows, cols):
        self.A_prev = None
        self.W = np.random.randn(rows, cols)*0.01
        self.b = np.zeros((rows, 1))
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        self.A_prev = A_prev
        A = self.W.dot(A_prev) + self.b
        return A

    def backward(self, dZ):
        dA_prev = self.W.T.dot(dZ)
        _, m = dZ.shape
        self.dW = dZ.dot(self.A_prev.T)/m
        self.db = np.sum(dZ, axis=1, keepdims=True)/m
        return dA_prev

    def update(self, lr):
        self.W -= lr*self.dW
        self.b -= lr*self.db


class CrossEntropy:
    def __init__(self):
        self.A_prev = None
        self.Y = None

    def forward(self, A_prev, Y):
        _, m = Y.shape
        self.Y = Y
        self.A_prev = A_prev
        A = -(Y.dot(np.log(A_prev).T) + (1 - Y).dot(np.log(1 - A_prev).T))/m
        return A

    def backward(self):
        return -(self.Y/self.A_prev - (1 - self.Y)/(1 - self.A_prev))


def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A


def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(Z))
    return A
