import sys, os
sys.path.append(os.curdir)
import numpy as np
from layers import *
import matplotlib.pyplot as plt


class NeuralNet:

    def __init__(self, input_size, output_size, hidden_layer_sizes):

        self.activation = {
            "ReLU": ReLU,
            "Sigmoid": Sigmoid,
            "TangentH": TangentH
        }

        # Add layers
        self.layers = []

        layer_sizes = [input_size] + \
            hidden_layer_sizes + \
            [output_size]

        L = len(layer_sizes)
        for i in range(1, L):
            self.add_layer(
                layer_sizes[i],
                layer_sizes[i - 1],
                "Sigmoid" if i == L - 1 else "ReLU"
            )

        # Layer for calculating cost
        self.last_layer = CrossEntropy()


    # Add Affine - Activation layer
    def add_layer(self, rows, cols, activation):
        self.layers.append(Affine(rows, cols))
        self.layers.append(self.activation[activation]())


    # Predict output by forward propagation
    # It does not propagate over last layer.
    def predict(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A


    # Caculate cost by forward propagation
    # It does propagate over last layer.
    def cost(self, X, Y):
        _, m = X.shape
        A = self.predict(X)
        cost = self.last_layer.forward(A, Y)
        return cost


    # Calculate prediction accuracy
    def accuracy(self, X, Y):
        A = self.predict(X)
        Y_hat = A > 0.5
        return np.mean(Y_hat == Y)


    # Caculate gradient by backward propagation
    # Gradient and parameter values are managed by layer class.
    def gradient(self, X, Y):
        dA = self.last_layer.backward()
        for layer in reversed(self.layers):
            dA = layer.backward(dA)


    def train(self, X, Y, num_iter=3000, lr=0.01):

        costs = []

        for j in range(num_iter):
            # Forward propagation
            cost = self.cost(X, Y)

            # Backward propagation
            self.gradient(X, Y)

            # Update parameters
            for layer in self.layers:
                if isinstance(layer, Affine):
                    layer.update(lr)

            # Show cost on console
            if j % 100 == 0:
                print(f"Cost for iteration {j}: {float(cost)}")
                costs.append(cost)

        # Draw cost graph
        plt.plot(np.squeeze(costs))
        plt.xlabel("Iterations (per hundreds)")
        plt.ylabel("Cost")
        plt.title("Cost curve")
        plt.show()

