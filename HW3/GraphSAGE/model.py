"""© 2018 Jianfei Gao All Rights Reserved"""
import math
import numpy as np
import torch
import torch.nn as nn


class GCN2(nn.Module):
    def __init__(self, aggregator, num_feats, num_hidden, num_labels, num_samples=5,
                 act='relu', dropout=0.5):
        r"""Initialize the class

        Args
        ----
        aggregator : Str
            Aggregator layer Name.
        num_feats : Int
            Size of input features for each node.
        num_hidden : Int
            Size of hidden embeddings for each node of each hidden layer.
        num_labels : Int
            Size of output labels.
        num_samples : Int
            Number of neighbor sampling for LSTM aggregator.
        act : Str
            Name of activation function.
        dropout : Int
            Dropout rate.

        """
        super(GCN2, self).__init__()

        self.gc1 = GraphConvolution(
            aggregator, None, num_feats, num_hidden, num_samples, act, dropout)
        self.gc2 = GraphConvolution(
            aggregator, self.gc1, num_hidden, num_labels, num_samples, None, dropout)

    def forward(self, adj_list, feats, batch):
        r"""

        Args
        ----
        adj_list : Dict
            Adjacent list of the graph.
        feats : torch.Tensor
            Input feature matrix of all nodes.
            It should be of shape (#nodes, #features).
        batch : [Int, Int, ...]
            A sequence of node ID in the forwarding batch.

        Returns
        -------
        probs : torch.Tensor
            Output label distribution matrix of all nodes before softmax.
            It should be of shape (#nodes, #labels).

        """
        return self.gc2(adj_list, feats, batch)

class GCN6(nn.Module):
    def __init__(self, aggregator, num_feats, num_hidden, num_labels, num_samples=5,
                 act='relu', dropout=0.5):
        r"""Initialize the class

        Args
        ----
        aggregator : Str
            Aggregator layer Name.
        num_feats : Int
            Size of input features for each node.
        num_hidden : Int
            Size of hidden embeddings for each node of each hidden layer.
        num_labels : Int
            Size of output labels.
        num_samples : Int
            Number of neighbor sampling for LSTM aggregator.
        act : Str
            Name of activation function.
        dropout : Int
            Dropout rate.

        """
        super(GCN6, self).__init__()

        self.gc1 = GraphConvolution(
            aggregator, None, num_feats, num_hidden, num_samples, act, dropout)
        self.gc2 = GraphConvolution(
            aggregator, self.gc1, num_hidden, num_hidden, num_samples, act, dropout)
        self.gc3 = GraphConvolution(
            aggregator, self.gc2, num_hidden, num_hidden, num_samples, act, dropout)
        self.gc4 = GraphConvolution(
            aggregator, self.gc3, num_hidden, num_hidden, num_samples, act, dropout)
        self.gc5 = GraphConvolution(
            aggregator, self.gc4, num_hidden, num_hidden, num_samples, act, dropout)
        self.gc6 = GraphConvolution(
            aggregator, self.gc5, num_hidden, num_labels, num_samples, None, dropout)

    def forward(self, adj_list, feats, batch):
        r"""

        Args
        ----
        adj_list : Dict
            Adjacent list of the graph.
        feats : torch.Tensor
            Input feature matrix of all nodes.
            It should be of shape (#nodes, #features).
        batch : [Int, Int, ...]
            A sequence of node ID in the forwarding batch.

        Returns
        -------
        probs : torch.Tensor
            Output label distribution matrix of all nodes before softmax.
            It should be of shape (#nodes, #labels).

        """
        return self.gc6(adj_list, feats, batch)

class MeanAggregator(nn.Module):
    def forward(self, feats):
        r"""Forwarding

        Args
        ----
        feats : torch.Tensor
            Features of all nodes to aggregate.
            It should be of shape (#nodes, #features).
        
        Returns
        -------
        embeds : torch.Tensor
            Output embedding matrix of aggregation.
            It should be of shape (#nodes, #embeddings).

        """
        raise NotImplementedError # missing

class LSTMAggregator(nn.Module):
    def __init__(self, num_feats, num_embeds):
        r"""Initialize the class

        Args
        ----
        num_feats : Int
            Size of input features for nodes to aggregate.
        num_embeds : Int
            Size of output embeddings of node aggregation.

        """
        super(LSTMAggregator, self).__init__()

        # allocate LSTM layer
        self.lstm = nn.LSTM(num_feats, num_feats)

    def forward(self, feats):
        r"""Forwarding

        Args
        ----
        feats : torch.Tensor
            Features of all nodes to aggregate.
            It should be of shape (#nodes, #features).
        
        Returns
        -------
        embeds : torch.Tensor
            Output embedding matrix of aggregation.
            It should be of shape (#nodes, #embeddings).

        """
        raise NotImplementedError # missing

class JanossyPool(nn.Module):
    def __init__(self, module, k):
        r"""Initialize the class

        Args
        ----
        module : nn.Module
            Part of neural network to apply Janossy pooling.
        k : Int
            Number of sampling.

        """
        super(JanossyPool, self).__init__()
        self.module = module
        self.k = k

    def forward(self, x):
        r"""Forwarding

        Args
        ----
        x : torch.Tensor
            Input tensor which supports permutation.
        
        Returns
        -------
        :x  torch.Tensor
            Output tensor which is permutation invariant of input.

        """
        indices = np.arange(x.size(0))
        np.random.shuffle(indices)
        return self.module(x[indices])

class GraphConvolution(nn.Module):
    def __init__(self, aggregator, features, num_feats, num_hidden, num_samples=5,
                 act='relu', dropout=0.5):
        r"""Initialize the class

        Args
        ----
        aggregator : Str
            Aggregator layer Name.
        features : nn.Module or Func
            Neural network layer which provides previous aggregated embeddings.
        num_feats : Int
            Size of input features for each node.
        num_hidden : Int
            Size of hidden embeddings for each node of each hidden layer.
        num_samples : Int
            Number of neighbor sampling for LSTM aggregator.
        act : Str
            Name of activation function.
        dropout : Int
            Dropout rate.

        """
        # parse arguments
        super(GraphConvolution, self).__init__()
        self.features = features
        self.num_samples = num_samples
        self.act_mode = act

        # allocate layers
        if aggregator == 'mean':
            self.aggregator = MeanAggregator()
        elif aggregator == 'lstm':
            self.aggregator = JanossyPool(LSTMAggregator(num_feats, num_feats), k=1)
        else:
            raise ValueError('unsupported \'aggregator\' argument')
        self.fc = nn.Linear(num_feats * 2, num_hidden)
        if self.act_mode is None:
            pass
        elif self.act_mode == 'relu':
            self.act = nn.ReLU()
            self.drop = nn.Dropout(dropout)
        else:
            raise ValueError('unsupported \'act\' argument')

    def forward(self, adj_list, feats, batch):
        r"""Forwarding

        Args
        ----
        adj_list : Dict
            Adjacent list of the graph.
        feats : torch.Tensor
            Input feature matrix of all nodes.
            It should be of shape (#nodes, #features).
        batch : [Int, Int, ...]
            A sequence of node ID in the forwarding batch.

        Returns
        -------
        embeds : torch.Tensor
            Output embedding matrix of all nodes.
            It should be of shape (#nodes, #embeddings).

        """
        raise NotImplementedError # missing


if __name__ == '__main__':
    pass
