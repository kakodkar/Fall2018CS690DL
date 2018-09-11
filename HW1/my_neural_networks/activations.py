"""© 2018 Jianfei Gao All Rights Reserved

- Original Version
    
    Author: I-Ta Lee
    Date: 10/1/2017

- Modified Version

    Author: Jianfei Gao
    Date: 08/28/2018

"""
import numpy as np
import torch


# +--------------------------------------------------------------------------------
# | Activation Functions
# +--------------------------------------------------------------------------------


EPSILON = 1e-14


def sigmoid(X):
    """Sigmoid Function (Sample Of Activation)

    Args
    ----
    X : tensor-like
        Shape should be (n_neurons, n_examples). 

    Returns
    -------
    Y : tensor-like
        A tensor where each element is the sigmoid of the X.

    Calculate sigmoid.
    
    """
    X_nexp = torch.exp(-X)
    return 1.0 / (1 + X_nexp)


def cross_entropy(X, y_1hot, epsilon=EPSILON):
    """Cross Entropy Loss

    Args
    ----
    X : tensor-like
        Softmax outputs. Shape should be (n_neurons, n_examples). 
    y_1hot : tensor-like
        1-hot-encoded labels. Shape should be (n_classes, n_examples). 
    epsilon : Float
        Log stability coefficence.

    Returns
    -------
    avg_ce : tensor-like
        Cross entropy loss (averaged).

    Cross entropy loss that assumes the input X is post-softmax, so this function only
    does negative loglikelihood. EPSILON is applied while calculating log.

    """
    raise NotImplementedError


def softmax(X):
    """Softmax

    Args
    ----
    X : tensor-like
        Shape should be (n_neurons, n_examples). 

    Returns
    -------
    Y : tensor-like
        Probability tensor derived from X. Shape should be (n_neurons, n_examples).

    Regular softmax.

    """
    raise NotImplementedError


def stable_softmax(X):
    """Softmax

    Args
    ----
    X : tensor-like
        Shape should be (n_neurons, n_examples). 

    Returns
    -------
    Y : tensor-like
        Probability tensor derived from X. Shape should be (n_neurons, n_examples).

    Numerically stable softmax.

    """
    raise NotImplementedError


def relu(X):
    """Rectified Linear Unit

    Args
    ----
    X : tensor-like
        Shape should be (n_neurons, n_examples). 

    Returns
    -------
    Y : tensor-like
        A tensor where the shape is the same as X but clamped on 0.
    
    Calculate ReLU.
    
    """
    raise NotImplementedError


# +--------------------------------------------------------------------------------
# | Feed Forward And Backpropagate On Fixed Architecture
# +--------------------------------------------------------------------------------


def feed_forward(X, weights, biases):
    """Forward pass
    
    Args
    ----
    X : tensor-like
        Input features of a neural network. Shape should be (n_neurons, n_examples).
    weights : List of tensor-like
        A list of weight tensors of layers of a neural network from input to output.
    biases : List of tensor-like
        A list of bias tensors of layers of a neural network from input to output.
        Each bias tensor should correspond to the tensor of the same position in
        weights.

    Returns
    -------
    outputs, act_outputs : 2 Lists of tensor-like
        "outputs" is a list of torch tensors. Each tensor is the Wx+b (weighted sum plus
        bias) of each layer in the shape (n_neurons, n_examples).
        "act_outputs" is also a list of torch tensors. Each tensor is the "activated"
        outputs of each layer in the shape(n_neurons, n_examples). If f(.) is the
        activation function, this should be f(ouptuts).

    For the fixed architecture, please read example_networks.py.

    """
    raise NotImplementedError
    return outputs, act_outputs


def backpropagation(outputs, act_outputs, X, y_1hot, weights, biases):
    """Backward pass

    Args
    ----
    outputs : List of tensor-like
        Get from feed_forward(). Shape of each tensor should be (n_neurons, n_examples).
    act_outputs : List of tensor-like
        Get from feed_forward(). Shape of each tensor should be (n_neurons, n_examples).
    X : tensor-like
        Input features of a neural network. Shape should be (n_neurons, n_examples).
    y_1hot : tensor-like
        Ideal output labels of a neural network. Shape should be (n_classes, n_examples).
    weights : List of tensor-like
        A list of weight tensors of layers of a neural network from input to output.
    biases : List of tensor-like
        A list of bias tensors of layers of a neural network from input to output.
        Each bias tensor should correspond to the tensor of the same position in
        weights.

    Returns
    -------
    weight_grads : List of tensor-like
        Gradients corresponding to each tensor in weights.
    bias_grads : List of tensor-like
        Gradients corresponding to each tensor in biases.

    For the fixed architecture, please read example_networks.py.

    """
    raise NotImplementedError
    return weight_grads, bias_grads