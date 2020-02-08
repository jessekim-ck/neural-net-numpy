import sys, os
sys.path.append(os.curdir)
import numpy as np
from layers import *
import matplotlib.pyplot as plt


class NeuralNet:

    def __init__(self, input_size, layers, loss="CrossEntropy"):
        """
        layer_configs = [
            {"linear": "Affine", "activation": "ReLU", "size": 5},
            ...,
            {"linear": "Affine", "activation": "Sigmoid", "size": 1}
        ]
        """

        self.linear = {
            "Affine": Affine
        }

        self.activation = {
            "ReLU": ReLU,
            "Sigmoid": Sigmoid,
            "TangentH": TangentH,
            "SoftMax": SoftMax
        }

        self.loss = {
            "CrossEntropy": CrossEntropy,
        }

        # Add layers
        self.layers = []

        prev_layer_size = None
        for i, layer in enumerate(layers):
            self.add_layer(
                linear = layer["linear"],
                rows = layer["size"],
                cols = prev_layer_size or input_size,
                activation = layer["activation"]
            )

            prev_layer_size = layer["size"]

        # Layer for calculating cost
        self.last_layer = self.loss[loss]()


    # Add Affine - Activation layer
    def add_layer(self, linear, rows, cols, activation):
        self.layers.append(self.linear[linear](rows, cols))
        self.layers.append(self.activation[activation]())


    def forward_propagate(self, X, Y):
        A = X
        for layer in self.layers:
            # print("Forwardpropagating layer " + str(layer))
            A = layer.forward(A)
        # print("Prediction: ")
        # print(A)
        cost = self.last_layer.forward(A, Y)
        accuracy = self.accuracy(A, Y)

        return cost, accuracy

    
    def accuracy(self, A, Y):
        Y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=0)

        return np.mean(Y_hat == Y)



    def backward_propagate(self, lr):
        dA = self.last_layer.backward()
        for layer in reversed(self.layers):
            # print("Backpropagating layer " + str(layer))
            dA = layer.backward(dA)
            # print("dA: ")
            # print(dA)
            if isinstance(layer, Affine):
                layer.update(lr)
                # print("Updated layer " + str(layer))
        return dA


    def train(self, X, Y, epochs=100, batch_size=1000, lr=0.01, verbose=True):

        costs = []
        accuracies = []
        _, train_size = X.shape
        iter_per_epoch = max(np.floor(train_size/batch_size), 1)
        max_iter = int(epochs*iter_per_epoch)

        for i in range(max_iter):

            batch = np.random.choice(train_size, batch_size)
            X_batch = X[:, batch]
            Y_batch = Y[:, batch]

            cost, accuracy = self.forward_propagate(X_batch, Y_batch)
            self.backward_propagate(lr)

            # Show cost on console
            if i % 100 == 0:
                costs.append(cost)
                accuracies.append(accuracy)
                print(f"Iteration {i} | cost: {float(cost):.6f}")

        if verbose:
            # Draw cost graph
            plt.plot(np.squeeze(zip(costs, accuracies)))
            plt.xlabel("Iterations (per hundreds)")
            plt.ylabel("Cost")
            plt.title("Cost curve")
            plt.show()



