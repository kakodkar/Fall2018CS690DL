"""© 2018 Jianfei Gao All Rights Reserved

Usage:
    main.py [--root <folder>] [--agg <name>] [--layer <N>] [--hidden <N>] [--lr <lr>]
            [--l2 <l2>] [--epoch <N>] [--clip <C>] [--save <path>] [--device <ID>] [--seed <S>]

Options:
    --root <folder>     Root folder of cora dataset [default: ./data/cora]
    --agg <name>        Aggregator to use [default: mean]
    --layer <N>         Number of graph convolutional layers [default: 2]
    --hidden <N>        Number of neurons of each hidden layer [default: 16]
    --lr <lr>           Learning rate [default: 0.01]
    --l2 <l2>           L2 regularization strength [default: 5e-4]
    --epoch <N>         Number of training epochs [default: 100]
    --clip <C>          Gradient clipping scalor
    --save <path>       Path to save model parameters [default: ./model.pt]
    --device <ID>       GPU card ID (not being specified as CPU) to work on
    --seed <S>          Random seed [default: 42]

"""
import docopt
from schema import Schema, Use, And, Or
import numpy as np
import torch

from data import CoraDataset
from model import GCN2, GCN6
from utils import random_seed, accuracy


def prepare_data(root, shuffle=False):
    r"""Prepare data for different usages

    Args
    ----
    root : Str
        Root folder of cora dataset.
    shuffle : Bool
        Randomly shuffle raw data.

    Returns
    -------
    dataset : CoraDataset
        Cora dataset.
    train_indices : np.array
        An array of points ID to train on.
    valid_indices : np.array
        An array of points ID to validate on.
    test_indices : np.array
        An array of points ID to test on.

    """
    # load dataset
    dataset = CoraDataset(root)

    # separate dataset indices
    num_nodes = len(dataset.pt_str2int)
    if shuffle:
        indices = np.random.permutation(num_nodes)
    else:
        indices = np.arange(num_nodes)
    train_indices = indices[1500:]
    valid_indices = indices[1000:1500]
    test_indices = indices[0:1000]

    return dataset, train_indices, valid_indices, test_indices

def prepare_model(aggregator, num_layers, num_feats, num_hidden, num_labels, lr, l2):
    r"""Prepare model

    Args
    ----
    aggregator : Str
        Name of aggregator.
    num_layers : Int
        Number of graph convolutional layers.
    num_feats : Int
        Number of input neurons.
    num_hidden : Int
        Number of hidden neurons.
    num_labels : Int
        Number of output neurons.
    lr : Float
        Learning rate.
    l2 : Float
        L2 regularization strength.

    Returns
    -------
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Training parameter optimizer.

    """
    GCNs = {2: GCN2, 6: GCN6}
    model = GCNs[num_layers](
        aggregator, num_feats, num_hidden, num_labels,
        num_samples=5, act='relu', dropout=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    return model, criterion, optimizer

def train(dataset, indices, model, criterion, optimizer, clip):
    r"""Train an epoch

    Args
    ----
    dataset : CoraDataset
        Cora dataset.
    indices : np.array
        An array of points ID to train on.
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Training parameter optimizer.
    clip : Float or None
        Gradient clipping scalor.

    """
    model.train()
    np.random.shuffle(indices)
    for i in range(len(indices) // 32):
        batch_indices = indices[i * 32:i * 32 + 32]
        optimizer.zero_grad()
        output = model(dataset.adj_list, dataset.feat_mx, batch_indices)
        loss = criterion(output, dataset.label_mx[batch_indices])
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        else:
            pass
        optimizer.step()

def evaluate(dataset, indices, model, criterion):
    r"""Evaluate

    Args
    ----
    dataset : CoraDataset
        Cora dataset.
    indices : np.array
        An array of points ID to evaluate on.
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.

    Returns
    -------
    loss : Float
        Averaged loss.
    acc : Float
        Averaged accuracy.

    """
    model.eval()
    output = model(dataset.adj_list, dataset.feat_mx, indices)
    loss = criterion(output, dataset.label_mx[indices])
    acc = accuracy(output, dataset.label_mx[indices])
    return float(loss.data.item()), float(acc)

def fit(dataset, train_indices, valid_indices,
        model, criterion, optimizer,
        num_epochs=100, clip=None, save='./model.pt', device=None):
    r"""Fit model parameters

    Args
    ----
    dataset : CoraDataset
        Cora dataset.
    train_indices : np.array
        An array of points ID to train on.
    valid_indices : np.array
        An array of points ID to validate on.
    model : torch.nn.Module
        Neural network model.
    criterion : torch.nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer
        Training parameter optimizer.
    num_epochs : Int
        Number of training epochs.
    clip : Float or None
        Gradient clipping scalor.
    save : Str
        Path to save best model parameters.
    device : Int or None
        GPU card to work on.

    """
    print('=' * 58)
    print("Train {} Epochs".format(num_epochs))
    print('-' * 58)

    # transfer to GPU
    if device is not None and torch.cuda.is_available():
        model = model.cuda(device)
        criterion = criterion.cuda(device)
        dataset.feat_mx = dataset.feat_mx.cuda(device)
        dataset.label_mx = dataset.label_mx.cuda(device)
    else:
        pass

    # first evaluation
    train_loss, train_acc = evaluate(dataset, train_indices, model, criterion)
    valid_loss, valid_acc = evaluate(dataset, valid_indices, model, criterion)

    best_loss = valid_loss
    torch.save(model.state_dict(), save)

    print("[{:>3d}] Train: {:.6f} ({:.3f}%), Valid: {:.6f} ({:.3f}%)".format(
            0, train_loss, train_acc, valid_loss, valid_acc))

    # train and evaluate
    for i in range(1, num_epochs + 1):
        train(dataset, train_indices, model, criterion, optimizer, clip)
        train_loss, train_acc = evaluate(dataset, train_indices, model, criterion)
        valid_loss, valid_acc = evaluate(dataset, valid_indices, model, criterion)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), save)
        else:
            pass

        print("[{:>3d}] Train: {:.6f} ({:.3f}%), Valid: {:.6f} ({:.3f}%)".format(
                i, train_loss, train_acc, valid_loss, valid_acc))

    print('=' * 58)
    print()


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    requirements = {
        '--hidden': Use(int),
        '--layer' : Use(int),
        '--lr'    : Use(float),
        '--l2'    : Use(float),
        '--epoch' : Use(int),
        '--clip'  : Or(None, Use(float)),
        '--device': Or(None, Use(int)),
        '--seed'  : Use(int),
        object    : object,
    }
    args = Schema(requirements).validate(args)

    seed = args['--seed']
    random_seed(seed)

    root = args['--root']
    dataset, train_indices, valid_indices, test_indices = prepare_data(root)

    aggregator = args['--agg']
    num_layers = args['--layer']
    num_feats = dataset.feat_mx.shape[1]
    num_hidden = args['--hidden']
    num_labels = dataset.label_mx.max().item() + 1
    lr = args['--lr']
    l2 = args['--l2']
    model, criterion, optimizer = prepare_model(
        aggregator, num_layers, num_feats, num_hidden, num_labels, lr, l2)

    num_epochs = args['--epoch']
    clip = args['--clip']
    save = args['--save']
    device = args['--device']
    fit(
        dataset, train_indices, valid_indices, model, criterion, optimizer,
        num_epochs, clip, save, device)

    model.load_state_dict(torch.load(save))
    test_loss, test_acc = evaluate(dataset, test_indices, model, criterion)
    print('=' * 24)
    print("Test Loss     : {:.6f}".format(test_loss))
    print("Test Accuracy : {:.3f}%".format(test_acc))
    print('=' * 24)