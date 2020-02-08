import numpy as np
import warnings
warnings.filterwarnings("error")


class ReLU:

    def __init__(self):
        self.A = None

    def forward(self, Z):
        A = np.fmax(0, Z)
        self.A = A
        return A

    def backward(self, dA):
        df = np.float_(self.A > 0)
        dZ = dA*df
        return dZ


class Sigmoid:

    def __init__(self):
        self.A = None

    def forward(self, Z):
        A = sigmoid(Z)
        self.A = A
        return A

    def backward(self, dA):
        df = self.A*(1 - self.A)
        dZ = dA*df
        return dZ


class TangentH:

    def __init__(self):
        self.A = None

    def forward(self, Z):
        A = tanh(Z)
        self.A = A
        return A

    def backward(self, dA):
        df = 1 - self.A**2
        dZ = dA*df
        return dZ


class SoftMax:

    def __init__(self):
        self.A = None

    def forward(self, Z):
        try:
            C = np.max(Z, axis=0, keepdims=True)
            Z = Z - C # Prevent Overflow
            A = np.exp(Z)/np.sum(np.exp(Z), axis=0, keepdims=True)
            self.A = A
            return A
        except RuntimeWarning as e:
            print(e)
            print(Z)

    def backward(self, dA):
        dZ = self.A*(1 + dA)
        
        assert(self.A.shape == dZ.shape)

        return dZ


class Affine:

    def __init__(self, rows, cols):
        self.A_prev = None
        self.W = np.random.randn(rows, cols)*0.01
        self.b = np.random.randn(rows, 1)*0.01
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        self.A_prev = A_prev
        Z = self.W.dot(A_prev) + self.b
        return Z

    def backward(self, dZ):
        _, m = dZ.shape
        dA_prev = self.W.T.dot(dZ)
        self.dW = dZ.dot(self.A_prev.T)/m
        self.db = np.sum(dZ, axis=1, keepdims=True)/m

        assert(dA_prev.shape == self.A_prev.shape)
        assert(self.dW.shape == self.W.shape)
        assert(self.db.shape == self.b.shape)

        return dA_prev

    def update(self, lr):
        # print("Update W. dW: ")
        # print(self.dW)
        self.W -= lr*self.dW
        # print("Updating b. db: ")
        # print(self.db)
        self.b -= lr*self.db


class CrossEntropy:
    def __init__(self):
        self.A_prev = None
        self.Y = None
        self.epsilon = 10**(-9)

    def forward(self, A_prev, Y):
        self.n, self.m = Y.shape
        self.Y = Y
        self.A_prev = A_prev

        if self.n == 1:
            A = -(Y.dot(np.log(self.A_prev).T) + (1 - Y).dot(np.log(1 - self.A_prev).T))/self.m
        else:
            A = -np.sum(Y*np.log(self.A_prev + self.epsilon))/self.m
        
        return A

    def backward(self):
        if self.n == 1:
            dA_prev =  -(self.Y/(self.A_prev) - (1 - self.Y)/(1 - self.A_prev))
        else:
            dA_prev = -np.divide(self.Y, self.A_prev + self.epsilon)

        assert(self.A_prev.shape == dA_prev.shape)
        
        return dA_prev


class MSE:

    def __init__(self):
        self.A_prev = None
        self.Y = None

    def forward(self, A_prev, Y):
        assert(A_prev.shape == Y.shape)
        self.A_prev = A_prev
        self.Y = Y
        _, m = Y.shape
        E = A_prev - Y
        A = E.dot(E.T)/(2*m)
        return A

    def backward(self):
        return self.A_prev - self.Y


def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A


def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(Z))
    return A
