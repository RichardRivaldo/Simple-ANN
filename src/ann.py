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
        # The full dataset, splitted features and target of the training dataset
        self.dataset = self.read_dataset(dataset)
        self.features = self.get_features()
        self.target = self.get_target()

        # Test set of the dataset
        self.test_set = self.get_test_set()

        # Initialize the number of nodes in each layers of the neural network
        self.layers_nodes = self.init_layers_nodes()

        # Initialized the weights and biases hyperparameters
        self.weights = self.init_weights()
        self.biases = self.init_biases()

        # Automatically fit and train the Neural Network to the dataset
        self.fit_train()

    # Read the dataset
    def read_dataset(self, dataset):
        dataset = pd.read_csv(dataset)
        return dataset

    # Get the features of the dataset in form of Numpy array
    def get_features(self, n_data=280):
        return self.dataset.iloc[:n_data, :-1].to_numpy()

    # Get the target of the dataset in form of Numpy array
    def get_target(self, n_data=280):
        target = self.dataset.iloc[:n_data, -1].to_numpy()
        return np.reshape(target, (len(target), 1))

    # Get the test set of the dataset in form of Numpy array
    def get_test_set(self):
        # Find the index to start
        start_idx = len(self.features)

        # Slice the rest of the dataset starting from the start index
        return self.dataset.iloc[start_idx:, :].to_numpy()

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

    # Avoid infinity by changing 0 to a very small value
    def change_zero(self, y_hat, y_hat_inv):
        y_hat = np.maximum(y_hat, 0.0000000001)
        y_hat_inv = np.maximum(y_hat_inv, 0.0000000001)

        return y_hat, y_hat_inv

    # Cross Entropy Cost Function
    # In the case of binary classification, this cost function should be used instead of MSE
    # ***The implementation is prone to overflow error and miscalculation***
    def cross_entropy(self, y, y_hat):
        # Calculate inverse
        y_inv = 1.0 - y
        y_hat_inv = 1.0 - y_hat

        # Avoid getting infinity by changing 0 values with very small number
        y_hat, y_hat_inv = self.change_zero(y_hat, y_hat_inv)

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
    def propagate_forward(self, X, y, cost="MSE"):
        # Initialize empty list attribute for activations and Z Value
        self.z_values = []
        self.activations = []

        # Get pairs of weights and biases
        weight_bias_pair = list(zip(self.weights, self.biases))

        # Calculate Z Value for Input-Hidden Layer nodes
        # Get the first weight and bias for the first pair of layers
        weight, bias = weight_bias_pair[0]

        # Dot product the features with the weight and add it with the bias
        z_value = X.dot(weight) + bias
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
            cost_error = self.mean_squared_error(y, y_hat)
        else:
            cost_error = self.cross_entropy(y, y_hat)

        return y_hat, cost_error

    # Backpropagation to calculate the gradients of the hyperparameters
    def propagate_backward(self, X, y, y_hat):
        # Calculate inverse
        y_inv = 1.0 - y
        y_hat_inv = 1.0 - y_hat

        # Avoid getting infinity by changing 0 values with very small number
        y_hat, y_hat_inv = self.change_zero(y_hat, y_hat_inv)

        # Find loss with respect to predicted Y
        dl_y_hat = np.divide(y_inv, y_hat_inv) - np.divide(y, y_hat)
        # Find loss with respect to the output of the last layer with derived sigmoid
        dl_sigmoid = y_hat * y_hat_inv
        dl_z2 = dl_y_hat * dl_sigmoid
        # Find loss with respect to the first activation value
        dl_A1 = dl_z2.dot(self.weights[-1].T)
        # Find loss with respect to the output of the hidden layer with derived ReLU
        dl_z1 = dl_A1 * self.derived_relu(self.z_values[0])

        # Initialize a list to contain gradients for each weights and biases
        self.weight_grad = []
        self.bias_grad = []

        # Calculate the gradient for the last weight and bias hyperparam
        dl_w2 = self.activations[0].T.dot(dl_z2)
        dl_b2 = np.sum(dl_z2, axis=0)

        # Calculate the gradient for the first weight and bias hyperparam
        dl_w1 = X.T.dot(dl_z1)
        dl_b1 = np.sum(dl_z1, axis=0)

        # Append all weight gradients
        self.weight_grad.append(dl_w1)
        self.weight_grad.append(dl_w2)

        # Append all bias gradients
        self.bias_grad.append(dl_b1)
        self.bias_grad.append(dl_b2)

    # Update the hyperparameters with certain learning rate
    def update_hyperparameters(self, lr=0.3):
        # Iterate over all elements in range of all weights
        # Since the biases will have same elements as weights,
        # the update can be applied in one loop
        for i in range(len(self.weights)):
            # Update the weight of the Neural Network
            self.weights[i] -= lr * self.weight_grad[i]
            # Update the bias of the Neural Network
            self.biases[i] -= lr * self.bias_grad[i]
