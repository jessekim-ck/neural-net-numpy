import numpy as np
import sys, os, warnings
warnings.filterwarnings("error")
sys.path.append(os.curdir)
from functions import *
from utils import *


class ReLU:

    def __init__(self):
        self.A = None

    def initialize(self, X):
        return self.forward(X[:, 0, None])

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

    def initialize(self, X):
        return self.forward(X[:, 0, None])

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

    def initialize(self, X):
        return self.forward(X[:, 0, None])

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

    def initialize(self, X):
        return self.forward(X[:, 0, None])

    def forward(self, Z):
        C = np.max(Z, axis=0, keepdims=True)
        Z = Z - C # Prevent Overflow
        A = np.exp(Z)/np.sum(np.exp(Z), axis=0, keepdims=True)
        self.A = A
        return A

    def backward(self, dA):
        dZ = self.A*(1 + dA)
        assert(self.A.shape == dZ.shape)
        return dZ


class Affine:

    def __init__(self, num_units):
        self.A_prev = None
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.num_units = num_units

    def initialize(self, X):
        # For convolution
        if X.ndim == 4:
            num_prev_units = X.shape[1]*X.shape[2]*X.shape[3]
        else:
            num_prev_units = X.shape[0]

        self.W = np.random.randn(self.num_units, num_prev_units)*0.01
        self.b = np.random.randn(self.num_units, 1)*0.01

        return self.forward(X[:, 0, None])

    def forward(self, A_prev):
        self.original_shape = A_prev.shape

        # For convolution
        if A_prev.ndim == 4:
            num_data = A_prev.shape[0]
            A_prev = A_prev.reshape(num_data, -1).T
        
        self.A_prev = A_prev
        Z = self.W.dot(A_prev) + self.b

        return Z

    def backward(self, dZ):
        num_data = dZ.shape[1]

        # Calculate gradients
        dA_prev = self.W.T.dot(dZ)
        self.dW = dZ.dot(self.A_prev.T)/num_data
        self.db = np.sum(dZ, axis=1, keepdims=True)/num_data

        # For convolution
        if len(self.original_shape) == 4:
            dA_prev = dA_prev.T.reshape(*self.original_shape)

        return dA_prev


class Convolution:

    def __init__(self, num_filters, filter_h, filter_w, stride=1, padding=0):
        self.W = None
        self.col_W = None
        self.b = None
        self.A_prev = None
        self.col_A_prev = None
        self.dW = None
        self.db = None
        self.stride = stride
        self.padding = padding
        self.num_filters = num_filters
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.num_channels = None

    def initialize(self, X):
        self.num_channels = X.shape[1]
        self.W = np.random.randn(self.num_filters, self.num_channels, self.filter_h, self.filter_w)*0.01
        self.b = np.random.randn(self.num_filters, 1, 1, 1)*0.01
        return self.forward(X[None, 0])
    
    def forward(self, A_prev):
        self.A_prev = A_prev

        # Get output shape
        num_data, num_channels, data_h, data_w = A_prev.shape
        Z_h = 1 + int((data_h + 2*self.padding - self.filter_h)/self.stride)
        Z_w = 1 + int((data_w + 2*self.padding - self.filter_w)/self.stride)

        # Convert 4D to 2D
        col_A_prev = im2col(self.A_prev, self.filter_h, self.filter_w, self.stride, self.padding)
        col_W = self.W.reshape(self.num_filters, -1).T
        self.col_A_prev = col_A_prev
        self.col_W = col_W

        col_Z = col_A_prev.dot(col_W) + self.b
        
        # Reshape to be (num_data, num_channels, Z_h, Z_w)
        Z = col_Z.reshape(num_data, Z_h, Z_w, -1).transpose(0, 3, 1, 2)

        return Z

    def backward(self, dZ):
        num_data = dZ.shape[0]

        # Reshape to be (num_data, Z_h, Z_w, num_channels)
        dZ = dZ.transpose(0, 2, 3, 1).reshape(-1, self.num_filters)

        # Calculate gradients :: should divide by num_data?
        self.dW = self.col_A_prev.T.dot(dZ)/num_data
        self.dW = self.dW.transpose(1, 0).reshape(*self.W.shape)
        self.db = np.sum(dZ, axis=0, keepdims=True)/num_data
        col_dA_prev = dZ.dot(self.col_W.T)
        dA_prev = col2im(col_dA_prev, self.A_prev.shape, self.filter_h, self.filter_w, self.stride, self.padding)

        return dA_prev


class Pooling:

    def __init__(self, pool_h, pool_w, stride=1, padding=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding
        self.Z = None

    def initialize(self, X):
        return self.forward(X[None, 0])
    
    def forward(self, Z):
        self.Z = Z
        num_data, num_channels, height, width = Z.shape
        A_h = int((height - self.pool_h)/self.stride + 1)
        A_w = int((width - self.pool_w)/self.stride + 1)

        col_Z = im2col(Z, self.pool_h, self.pool_w, self.stride, self.padding)
        col_Z = col_Z.reshape(-1, self.pool_h*self.pool_w)

        self.arg_max = np.argmax(col_Z, axis=1)
        A = np.max(col_Z, axis=1)
        A = A.reshape(num_data, A_h, A_w, num_channels).transpose(0, 3, 1, 2)

        return A

    def backward(self, dZ):
        dZ = dZ.transpose(0, 2, 3, 1)

        pool_size = self.pool_h*self.pool_w
        dmax = np.zeros((dZ.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dZ.flatten()
        dmax = dmax.reshape(dZ.shape + (pool_size, ))

        col_dA = dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2], -1)
        dA = col2im(col_dA, self.Z.shape, self.pool_h, self.pool_w, self.stride, self.padding)

        return dA


class CrossEntropy:

    def __init__(self):
        self.A_prev = None
        self.Y = None
        self.epsilon = 10**(-9)

    def forward(self, A_prev, Y):
        self.n, self.m = Y.shape
        self.Y = Y
        self.A_prev = A_prev + self.epsilon
        if self.n == 1: # case binary classification
            A = -(Y.dot(np.log(self.A_prev).T) + (1 - Y).dot(np.log(1 - self.A_prev).T))/self.m
        else: # case multiclass classification
            A = -np.sum(Y*np.log(self.A_prev))/self.m        
        return A

    def backward(self):
        if self.n == 1: # case binary classification
            dA_prev =  -(self.Y/(self.A_prev) - (1 - self.Y)/(1 - self.A_prev))
        else: # case multiclass classification
            dA_prev = -np.divide(self.Y, self.A_prev)
        return dA_prev


class MeanSquaredError:

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
