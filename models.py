import sys, os
sys.path.append(os.curdir)
import numpy as np
from layers import *
import matplotlib.pyplot as plt


class NeuralNet:

    def __init__(self, input_size, layers, loss="CrossEntropy"):
        """
        layers = [
            {"linear": "Affine", "activation": "ReLU", "size": 5},
            ...,
            {"linear": "Affine", "activation": "Sigmoid", "size": 1}
        ]
        """

        # List of linear layer class
        self.linear = {
            "Affine": Affine
        }

        # List of activation layer class
        self.activation = {
            "ReLU": ReLU,
            "Sigmoid": Sigmoid,
            "TangentH": TangentH,
            "SoftMax": SoftMax
        }

        # List of loss layer class
        self.loss = {
            "CrossEntropy": CrossEntropy,
            "MeanSquaredError": MeanSquaredError
        }

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

        self.last_layer = self.loss[loss]()

    # Add linear-activation layer set.
    def add_layer(self, linear, rows, cols, activation):
        self.layers.append(self.linear[linear](rows, cols))
        self.layers.append(self.activation[activation]())

    # Predict output given data.
    def predict(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    # Calculate accuracy given prediction.
    # Presume classification, one-hot-encoding.
    def accuracy(self, A, Y):
        Y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=0)

        return np.mean(Y_hat == Y)

    # Forward propagation.
    # You should call this method before backward propagation.
    def forward_propagate(self, X, Y):
        A = self.predict(X)
        cost = self.last_layer.forward(A, Y)
        accuracy = self.accuracy(A, Y)

        return cost, accuracy

    # Backward propagation.
    # This updates gradients for each layer class.
    def backward_propagate(self):
        dA = self.last_layer.backward()
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
        return dA

    # Update parameters by its gradient.
    # Need to be componentized for various optimization algorithm.
    def update_params(self, lr):
        for layer in reversed(self.layers):
            if isinstance(layer, Affine):
                layer.W -= lr*layer.dW
                layer.b -= lr*layer.db

    def train(self, X, Y, epochs=20, batch_size=100, lr=0.01, verbose=True):

        costs = []
        accuracies = []

        _, train_size = X.shape
        iter_per_epoch = max(np.floor(train_size/batch_size), 1)
        max_iter = int(epochs*iter_per_epoch)

        for i in range(max_iter):

            print(f"--Iteration: {i + 1}", end="\r")

            # Sample batch
            batch = np.random.choice(train_size, batch_size)
            X_batch = X[:, batch]
            Y_batch = Y[:, batch]

            # Forward propagation
            self.forward_propagate(X_batch, Y_batch)

            # Backward propagation
            self.backward_propagate()

            # Update parameters
            self.update_params(lr)

            # Record cost/accuracy per epochs
            if i % iter_per_epoch == 0:
                cost, accuracy = self.forward_propagate(X, Y)
                costs.append(cost)
                accuracies.append(accuracy)
                print(f"Epoch {int(i/iter_per_epoch)} | cost: {float(cost):.6f} | accuracy: {float(accuracy):.4f}")

        # Draw cost/accuracy graph
        if verbose:
            plt.plot(np.squeeze(costs), label="cost")
            plt.plot(np.squeeze(accuracies), label="accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Cost")
            plt.title("Cost curve")
            plt.legend()
            plt.show()
