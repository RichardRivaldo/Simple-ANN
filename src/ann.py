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
        # The full dataset, splitted features and target of the dataset
        self.dataset = self.read_dataset(dataset)
        self.features = self.get_features()
        self.target = self.get_target()

        # Initialize the number of nodes in each layers of the neural network
        self.layers_nodes = self.init_layers_nodes()

        # Initialized the weights and biases hyperparameters
        self.weights = self.init_weights()
        self.biases = self.init_biases()

    # Read the dataset
    def read_dataset(self, dataset):
        dataset = pd.read_csv(dataset)
        return dataset

    # Get the features of the dataset in form of Numpy array
    def get_features(self):
        return self.dataset.iloc[:, :-1].to_numpy()

    # Get the target of the dataset in form of Numpy array
    def get_target(self):
        return self.dataset.iloc[:, -1].to_numpy()

    # Initialize the number of nodes in the network's layers
    def init_layers_nodes(self):
        # Set the properties of the neural network
        # Set number of nodes in every layers
        # The input layer will have number of features nodes, the output will be 1
        # and the hidden layer will have half of the number of features nodes ceiled
        num_of_features = self.features.shape[1]
        return [num_of_features, int(np.ceil(num_of_features / 2)), 1]

    # Initialize random weights
    # Use Numpy Randn to initialize random normal-distributed values to array with given dims
    def init_weights(self):
        # Initialize weights except for output layer
        # The neural networks will be dense, meaning that each hidden layer nodes will have
        # connections to each input layers. Therefore, if num_of_features = n,
        # the dims of the weights will be ceil(n/2) x n for input-hidden layer
        # and number of nodes in the output layer (1) x (n/2) for hidden-output layer
        random_weights = [
            np.random.randn(m, n)
            for m, n in zip(self.layers_nodes[1:], self.layers_nodes[:-1])
        ]

        # Transpose the weights to fit the dot product shape
        transposed_weights = []
        for weights in random_weights:
            transposed_weights.append(weights.T)

        return transposed_weights

    # Initialize random biases
    # Use Numpy Randn to initialize random normal-distributed values to array with given dims
    def init_biases(self):
        # Initialize only biases for non-input layer nodes
        # The biases will be in form of 1 x 7 for hidden layers, and constant for output layer
        random_biases = [np.random.randn(layers, 1) for layers in self.layers_nodes[1:]]

        # Reshape the biases to fit the sum operation later on
        reshaped_biases = []
        for biases in random_biases:
            reshaped_biases.append(biases.reshape(biases.shape[0]))

        return reshaped_biases

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
        return 1.0 / (1.0 + np.exp(-x))

    # Sigmoid Derivatives
    def derived_sigmoid(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    # Mean Squared Error (MSE)
    def mean_squared_error(self, y, y_hat):
        return np.mean((y_hat - y) ** 2)

    # Cross Entropy Cost Function
    # In the case of binary classification, this cost function should be used instead of MSE
    # ***The implementation is prone to overflow error and miscalculation***
    def cross_entropy(self, y, y_hat):
        # Calculate inverse
        y_inv = 1.0 - y
        y_hat_inv = 1.0 - y_hat

        # Avoid getting infinity by changing 0 values with very small number
        y_hat_inv = np.maximum(y_hat_inv, 0.0000000001)
        y_hat = np.maximum(y_hat_inv, 0.0000000001)

        # Calculate the CE Cost
        one_per_n = -1 / len(y)
        ce_cost = one_per_n * (
            np.sum(
                np.multiply(np.log(y_hat), y) + np.multiply((y_inv), np.log(y_hat_inv))
            )
        )

        return ce_cost

    # Feed Forward Propagation
    # Default cost used: MSE
    def propagate_forward(self, cost="MSE"):
        # Initialize empty list attribute for activations and Z Value
        self.z_values = []
        self.activations = []

        # Get pairs of weights and biases
        weight_bias_pair = list(zip(self.weights, self.biases))

        # Calculate Z Value for Input-Hidden Layer nodes
        # Get the first weight and bias for the first pair of layers
        weight, bias = weight_bias_pair[0]

        # Dot product the features with the weight and add it with the bias
        z_value = self.features.dot(weight) + bias
        # Find the activation value of the first pair of layers with ReLU function
        act_value = self.relu(z_value)
        # Append the Z and activation value to the list
        self.z_values.append(z_value)
        self.activations.append(act_value)

        # Calculate Z Value for Hidden Layer-Output Layer nodes
        # Get the second weight and bias for the second pair of layers
        weight, bias = weight_bias_pair[1]

        # Dot product the features with the weight and add it with the bias
        z_value = act_value.dot(weight) + bias
        # Find the activation value of the first pair of layers with Sigmoid function
        y_hat = self.sigmoid(z_value)
        # Append the Z and activation value to the list
        self.z_values.append(z_value)
        self.activations.append(y_hat)

        # Calculate the error cost
        if cost == "MSE":
            cost_error = self.mean_squared_error(self.target, y_hat)
        else:
            cost_error = self.cross_entropy(self.target, y_hat)

        return y_hat, cost_error
