# Implementasikan ketiga bagian dari ANN:
# Forward Propagation
# Back Propagation
# Updating Weight
# Algoritma yang dibuat, minimal dapat:
# Menerima optimizer Stochastic Gradient Descent (SGD)
# Menerima loss function MSE
# Dapat menerima activation function sigmoid dan relu
# Mengkategorikan label binary (True False)

# Libraries
import pandas as pd
import numpy as np


class ArtificialNeuralNetwork:
    # Constructor of the ANN
    def __init__(self, dataset):
        # The dataset, splitted feature and target of the dataset
        self.dataset = self.read_dataset(dataset)
        self.features = self.get_features()
        self.target = self.get_target()

        # Set the properties of the neural network
        # Set number of nodes in every layers
        # The input layer will have number of features nodes, the output will be 1
        # and the hidden layer will have half of the number of features nodes ceiled
        num_of_features = self.features.shape[1]
        self.layers_nodes = [num_of_features, int(np.ceil(num_of_features / 2)), 1]

        # Initialized the weights and biases hyperparameters
        # Use Numpy Randn to initialize random normal-distributed values to array with given dims

        # Initialize weights except for output layer
        # The neural networks will be dense, meaning that each hidden layer nodes will have
        # connections to each input layers. Therefore, if num_of_features = n,
        # the dims of the weights will be ceil(n/2) x n for input-hidden layer
        # and number of nodes in the output layer (1) x (n/2) for hidden-output layer
        self.weights = [
            np.random.randn(m, n)
            for m, n in zip(self.layers_nodes[1:], self.layers_nodes[:-1])
        ]
        # Initialize only biases for non-input layer nodes
        self.biases = [np.random.randn(layers, 1) for layers in self.layers_nodes[1:]]

    # Read the dataset
    def read_dataset(self, dataset):
        dataset = pd.read_csv(dataset)
        return dataset

    # Get the features of the dataset
    def get_features(self):
        return self.dataset.iloc[:, :-1]

    # Get the target of the dataset
    def get_target(self):
        return self.dataset.iloc[:, -1]

    # ReLU activation function
    # Return 0 if the value is less than 0, else return the value
    def relu(self, x):
        return np.maximum(0.0, x)

    # ReLU Derivatives
    def derived_relu(self, x):
        return 1 * (x > 0)

    # Sigmoid activation function
    # Return 0 if the value is 0.5, else use the exp equation
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sigmoid Derivatives
    def derived_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
