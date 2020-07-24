import sys, os
sys.path.append(os.curdir)
import numpy as np
from models import *
import time
from PIL import Image


# Show image
def img_show(img):
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()

# Load mnist data
def load_mnist(test=False, flatten=False):
    if test:
        path = "mnist/mnist_test.csv"
    else:
        path = "mnist/mnist_train.csv"
    mnist = np.genfromtxt(path, delimiter=",")

    X = mnist[1:, 1:]
    Y = mnist[1:, 0]
    Y = np.array([[float(i == y) for i in range(10)] for y in Y]).T

    if flatten == False:
        num_data = X.shape[0]
        X = X.reshape(num_data, 1, 28, 28)
    else:
        X = X.T

    X /= 255 # Normalize data

    return X, Y

# Load train data
print("Loading data...")
X_train, Y_train = load_mnist(test=True)
for i in [3, 2, 1]:
    print(f"Learning start: {i}", end="\r")
    time.sleep(0.5)

network = NeuralNet()

# Add layers to the network
A = X_train
A = network.add_layer("Convolution", A, num_filters=4, filter_h=3, filter_w=3)
A = network.add_layer("ReLU", A)
A = network.add_layer("Pooling", A, pool_h=3, pool_w=3, stride=3)
A = network.add_layer("Affine", A, num_units=15)
A = network.add_layer("ReLU", A)
A = network.add_layer("Affine", A, num_units=10)
A = network.add_layer("SoftMax", A)
network.set_loss("CrossEntropy")


# Learning rate of 0.3 worked well on [64, 64, 64, 10] layers.
# Lesser learning rate didn't work.
network.train(X_train, Y_train, epochs=15, batch_size=100, lr=0.3, verbose=False)

# Load test data
X_test, Y_test = load_mnist(test = True)

# Get test accuracy
cost, accuracy = network.forward_propagate(X_test, Y_test)
print("\nTest accuracy: " + str(accuracy))
print("Learning Completed!")

predict = network.predict(X_test)

wrong_idx = (np.argmax(predict, axis=0) != np.argmax(Y_test, axis=0))


print("Malpredicted example:")
for i, wrong in enumerate(wrong_idx):
    if wrong == True:
        img_show(X_test[i, 0, :, :]*255)
        show_more = input("Wanna see more? Y/N: ")
        if show_more == "Y":
            pass
        else:
            break


print("\nBye!")
