"""© 2018 Jianfei Gao All Rights Reserved

- Original Version
    
    Author: I-Ta Lee
    Date: 10/1/2017

- Modified Version

    Author: Jianfei Gao
    Date: 08/28/2018

"""
import logging
import math
import numpy as np
import torch
from copy import deepcopy
from collections import OrderedDict

from .activations import relu, stable_softmax, cross_entropy
from .activations import feed_forward, backpropagation

class AutogradNeuralNetwork:
    """Implementation that uses torch.autograd

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
        # declare weights and biases
        if gpu_id == -1:
            self.weights = [torch.autograd.Variable(torch.FloatTensor(j, i),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.autograd.Variable(torch.FloatTensor(i, 1),
                                requires_grad=True)
                           for i in self.shape[1:]]
        else:
            self.weights = [torch.autograd.Variable(torch.randn(j, i).cuda(gpu_id),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.autograd.Variable(torch.randn(i, 1).cuda(gpu_id),
                                requires_grad=True)
                           for i in self.shape[1:]]
        # initialize weights and biases
        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases

            Initialize self.weights and self.biases with
            Gaussian where the std is 1 / sqrt(n_neurons)
        """
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            stdv = 1. / math.sqrt(w.size(1))
            # in-place random
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

    def _feed_forward(self, X):
        """Forward pass

        Args:
            X: (n_neurons, n_examples)

        Returns:
            (outputs, act_outputs).

            "outputs" is a list of torch tensors. Each tensor is the Wx+b (weighted sum plus bias)
            of each layer in the shape (n_neurons, n_examples).

            "act_outputs" is also a list of torch tensors. Each tensor is the "activated" outputs
            of each layer in the shape(n_neurons, n_examples). If f(.) is the activation function,
            this should be f(ouptuts).
        """
        return feed_forward(X, self.weights, self.biases)

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        # X_t_train = X_t           # forget to transpose
        # y_1hot_t_train = y_1hot_t # forget to transpose
        X_t_train = X.t()
        y_1hot_t_train = y_1hot.t()
        
        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        loss = cross_entropy(act_outputs[-1], y_1hot_t_train)

        # backward
        loss.backward()

        # update weights and biases
        for w, b in zip(self.weights, self.biases):
            w.data = w.data - (learning_rate * w.grad.data)
            b.data = b.data - (learning_rate * b.grad.data)
            w.grad.data.zero_()
            b.grad.data.zero_()
        # return loss.data[0] # Use for pytorch 0.3 or before
        return loss.item()
    
    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()
        outputs, act_outputs = self._feed_forward(X_t)
        loss = cross_entropy(act_outputs[-1], y_1hot_t)
        # return loss.data[0] # Use for pytorch 0.3 or before
        return loss.item()

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        outputs, act_outputs = self._feed_forward(X.t())
        return torch.max(act_outputs[-1], 0)[1]


class BasicNeuralNetwork:
    """Implementation using only torch.Tensor

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
        # declare weights and biases
        if gpu_id == -1:
            self.weights = [torch.FloatTensor(j, i)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.FloatTensor(i, 1)
                           for i in self.shape[1:]]
        else:
            self.weights = [torch.randn(j, i).cuda(gpu_id)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.randn(i, 1).cuda(gpu_id)
                           for i in self.shape[1:]]

        # initialize weights and biases
        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases

            Initialize self.weights and self.biases with
            Gaussian where the std is 1 / sqrt(n_neurons)
        """
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            stdv = 1. / math.sqrt(w.size(1))
            # in-place random
            w.uniform_(-stdv, stdv)
            b.uniform_(-stdv, stdv)

    def _feed_forward(self, X):
        """Forward pass

        Args:
            X: (n_neurons, n_examples)
        
        Returns:
            (outputs, act_outputs).

            "outputs" is a list of torch tensors. Each tensor is the Wx+b (weighted sum plus bias)
            of each layer in the shape (n_neurons, n_examples).

            "act_outputs" is also a list of torch tensors. Each tensor is the "activated" outputs
            of each layer in the shape(n_neurons, n_examples). If f(.) is the activation function,
            this should be f(ouptuts).
        """
        return feed_forward(X, self.weights, self.biases)

    def _backpropagation(self, outputs, act_outputs, X, y_1hot):
        """Backward pass

        Args:
            outputs: (n_neurons, n_examples). get from _feed_forward()
            act_outputs: (n_neurons, n_examples). get from _feed_forward()
            X: (n_features, n_examples). input features
            y_1hot: (n_classes, n_examples). labels
        """
        weight_grads, bias_grads = backpropagation(outputs, act_outputs, X, y_1hot, self.weights, self.biases)

        # apply gradient
        for w, b, w_grad, b_grad in zip(self.weights, self.biases, weight_grads, bias_grads):
            w.grad = w_grad
            b.grad = b_grad

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        # X_t_train = X_t           # forget to transpose
        # y_1hot_t_train = y_1hot_t # forget to transpose
        X_t_train = X.t()
        y_1hot_t_train = y_1hot.t()
        
        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        loss = cross_entropy(act_outputs[-1], y_1hot_t_train)

        # backward
        self._backpropagation(outputs, act_outputs, X_t_train, y_1hot_t_train)

        # update weights and biases
        for w, b in zip(self.weights, self.biases):
            w.data = w.data - (learning_rate * w.grad.data)
            b.data = b.data - (learning_rate * b.grad.data)
            w.grad.data.zero_()
            b.grad.data.zero_()
        # return loss.data[0] # Use for pytorch 0.3 or before
        return loss.item()

    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()
        outputs, act_outputs = self._feed_forward(X_t)
        loss = cross_entropy(act_outputs[-1], y_1hot_t)
        return loss

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        outputs, act_outputs = self._feed_forward(X.t())
        return torch.max(act_outputs[-1], 0)[1]
