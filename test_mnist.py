import sys, os
sys.path.append(os.curdir)
import numpy as np
from models import *


def load_mnist(test):
    if test:
        path = "mnist/mnist_test.csv"
    else:
        path = "mnist/mnist_train.csv"
    mnist = np.genfromtxt(path, delimiter=",")
    X = (mnist[1:, 1:].T)/255
    Y = mnist[1:, 0]

    # one-hot-encoding
    Y = np.array([[float(i == y) for i in range(10)] for y in Y]).T

    return X, Y


X_train, Y_train = load_mnist(test = False)
input_size, _ = X_train.shape
layer_sizes = [64, 10]
layers = [
    {
        "linear": "Affine",
        "activation": "SoftMax" if i == len(layer_sizes) - 1 else "Sigmoid",
        "size": size
    }
    for i, size in enumerate(layer_sizes)
]
network = NeuralNet(input_size, layers)
network.train(X_train, Y_train)


X_test, Y_test = load_mnist(test = True)
cost, accuracy = network.accuracy(X_test, Y_test)
print("\nTest accuracy: " + str(accuracy))
print("Learning Completed!")
