#!/usr/bin/env python3
"""Defines a deep neural network with gradient descent."""

import numpy as np


class DeepNeuralNetwork:
    """Deep neural network performing binary classification."""

    def __init__(self, nx, layers):
        """Initialize the network with nx features and a list of layer sizes."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            prev = nx if i == 0 else layers[i - 1]
            self.__weights['W' + str(i + 1)] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            )
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Return the number of layers in the network."""
        return self.__L

    @property
    def cache(self):
        """Return the cache of intermediary values."""
        return self.__cache

    @property
    def weights(self):
        """Return the weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network."""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            z = np.matmul(W, A_prev) + b
            self.__cache['A' + str(i)] = 1 / (1 + np.exp(-z))
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculate the logistic regression cost of the model."""
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return np.sum(loss) / m

    def evaluate(self, X, Y):
        """Evaluate the network's predictions and return them with the cost."""
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Perform one pass of gradient descent on the network."""
        m = Y.shape[1]
        weights = self.__weights.copy()
        dz = cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            dw = np.matmul(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            if i > 1:
                dz = np.matmul(weights['W' + str(i)].T, dz) * (
                    A_prev * (1 - A_prev)
                )
            self.__weights['W' + str(i)] = (
                weights['W' + str(i)] - alpha * dw
            )
            self.__weights['b' + str(i)] = (
                weights['b' + str(i)] - alpha * db
            )
