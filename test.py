import sys, os
sys.path.append(os.curdir)
import numpy as np
from models import *


def create_linear():
    X = np.random.random((2, 5))
    W = np.array([[3, 1]])
    Y = W.dot(X) > 1
    return X, Y

X_train, Y_train = create_linear()
network = NeuralNet(2, 1, [3, 3, 3, 3, 3])
network.train(X_train, Y_train, num_iter=10000, lr=0.001)
accuracy = network.accuracy(X_train, Y_train)

print(f"\nAccuracy: {accuracy}\n")
