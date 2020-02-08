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
    Y = np.array([[float(i == y) for i in range(10)] for y in Y]).T
    return X, Y

# Load train data
X_train, Y_train = load_mnist(test=False)

# Get NueralNet object
input_size, _ = X_train.shape
layer_sizes = [64, 64, 64, 10]
layers = [
    {
        "linear": "Affine",
        "activation": "SoftMax" if i == len(layer_sizes) - 1 else "ReLU",
        "size": size
    }
    for i, size in enumerate(layer_sizes)
]
network = NeuralNet(input_size, layers)

# Learning rate of 0.3 worked well on [64, 64, 64, 10] with sigmoid activation.
# Lesser learning rate didn't work.
network.train(X_train, Y_train, epochs=30, batch_size=100, lr=0.3, verbose=True)

# Load test data
X_test, Y_test = load_mnist(test = True)

# Get test accuracy
cost, accuracy = network.forward_propagate(X_test, Y_test)
print("\nTest accuracy: " + str(accuracy))
print("Learning Completed!")
