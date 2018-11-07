"""© 2018 Jianfei Gao All Rights Reserved

Usage:
    main.py [--root <folder>] [--layer <N>] [--hidden <N>] [--lr <lr>] [--l2 <l2>]
            [--epoch <N>] [--save <path>] [--device <ID>] [--seed <S>]

Options:
    --root <folder>     Root folder of cora dataset [default: ./data/cora]
    --layer <N>         Number of graph convolutional layers [default: 2]
    --hidden <N>        Number of neurons of each hidden layer [default: 16]
    --lr <lr>           Learning rate [default: 0.01]
    --l2 <l2>           L2 regularization strength [default: 5e-4]
    --epoch <N>         Number of training epochs [default: 100]
    --save <path>       Path to save model parameters [default: ./model.pt]
    --device <ID>       GPU card ID (not being specified as CPU) to work on
    --seed <S>          Random seed [default: 42]

"""
import docopt
from schema import Schema, Use, And, Or
import numpy as np
import torch

from data import CoraDataset
from model import VariableGCN
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

def prepare_model(num_layers, num_feats, num_hidden, num_labels, lr, l2):
    r"""Prepare model

    Args
    ----
    num_feats : Int
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
    model = VariableGCN(num_feats, num_hidden, num_labels, num_layers, dropout=0)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    return model, criterion, optimizer

def train(dataset, indices, model, criterion, optimizer):
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

    """
    model.train()
    optimizer.zero_grad()
    output = model(dataset.adj_mx, dataset.feat_mx)
    loss = criterion(output[indices], dataset.label_mx[indices])
    loss.backward()
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
    output = model(dataset.adj_mx, dataset.feat_mx)
    loss = criterion(output[indices], dataset.label_mx[indices])
    acc = accuracy(output[indices], dataset.label_mx[indices])
    return float(loss.data.item()), float(acc)

def fit(dataset, train_indices, valid_indices,
        model, criterion, optimizer,
        num_epochs=100, save='./model.pt', device=None):
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
        dataset.adj_mx = dataset.adj_mx.cuda(device)
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
        train(dataset, train_indices, model, criterion, optimizer)
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

def check_mixing(dataset, indices, model):
    r"""Evaluate

    Args
    ----
    dataset : CoraDataset
        Cora dataset.
    indices : np.array
        An array of points ID to evaluate on.
    model : torch.nn.Module
        Neural network model.

    """
    model.eval()
    output = model(dataset.adj_mx, dataset.feat_mx)
    embeds = torch.nn.functional.normalize(model.embeds, p=2, dim=1)
    check_list = [0, 1, 2, 3, 4, 5]
    for i in range(len(check_list)):
        for j in range(i + 1, len(check_list)):
            print(((embeds[i] - embeds[j]) ** 2).sum())

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    requirements = {
        '--hidden': Use(int),
        '--layer' : Use(int),
        '--lr'    : Use(float),
        '--l2'    : Use(float),
        '--epoch' : Use(int),
        '--device': Or(None, Use(int)),
        '--seed'  : Use(int),
        object    : object,
    }
    args = Schema(requirements).validate(args)

    seed = args['--seed']
    random_seed(seed)

    root = args['--root']
    dataset, train_indices, valid_indices, test_indices = prepare_data(root)

    num_layers = args['--layer']
    num_feats = dataset.feat_mx.shape[1]
    num_hidden = args['--hidden']
    num_labels = dataset.label_mx.max().item() + 1
    lr = args['--lr']
    l2 = args['--l2']
    model, criterion, optimizer = prepare_model(
        num_layers, num_feats, num_hidden, num_labels, lr, l2)

    num_epochs = args['--epoch']
    save = args['--save']
    device = args['--device']
    fit(dataset, train_indices, valid_indices, model, criterion, optimizer,
        num_epochs, save, device)

    model.load_state_dict(torch.load(save))
    test_loss, test_acc = evaluate(dataset, test_indices, model, criterion)
    print('=' * 24)
    print("Test Loss     : {:.6f}".format(test_loss))
    print("Test Accuracy : {:.3f}%".format(test_acc))
    print('=' * 24)