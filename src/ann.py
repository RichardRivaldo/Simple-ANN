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
