"""© 2018 Jianfei Gao All Rights Reserved

Usage:
    main.py [--folder <dir>] [--batch-size <bsz>] [--test-batch-size <bsz>]
            [--epochs <N>] [--lr <lr>] [--device <N>]
            [--resume <path>] [--save <path>] [--seed <S>]

Data Options:
    --folder <dir>              Directory to load dataset [default: mnist]
    --batch-size <bsz>          Training batch size [default: 256]
    --test-batch-size <bsz>     Test (validation) batch size [default: 1000]

Training Options:
    --epochs <N>                Number of epochs to train [default: 100]
    --lr <lr>                   Learning rate [default: 0.001]
    --device <N>                GPU (CUDA) device to train on

IO Options:
    --save <path>               Path to save model parameters
    --resume <path>             Path to resume model parameters

Random Options:
    --seed <S>                  Random seed [default: 1234567890]

"""
import time
import torch
import torch.nn as Module
from torchvision import datasets as Dataset
from torchvision import transforms as Transform
import torch.optim as Optimizer


# Neural Network Definition
class ShallowNet(Module.Module):
    """A Shallow Convolutional Neural Network"""
    def __init__(self):
        """Initialize the Class"""
        # parse arguments
        super(ShallowNet, self).__init__()

        # allocate layer 1
        self.conv1 = Module.Conv2d(1, 10, kernel_size=5)
        self.bn1   = Module.BatchNorm2d(10)
        self.relu1 = Module.ReLU()
        self.pool1 = Module.MaxPool2d(kernel_size=2)

        # allocate layer 2
        self.conv2 = Module.Conv2d(10, 20, kernel_size=5)
        self.bn2   = Module.BatchNorm2d(20)
        self.relu2 = Module.ReLU()
        self.pool2 = Module.MaxPool2d(kernel_size=2)
        self.drop2 = Module.Dropout2d(0.5)

        # allocate layer 3
        self.fc3   = Module.Linear(320, 50)
        self.bn3   = Module.BatchNorm1d(50)
        self.relu3 = Module.ReLU()

        # allocate layer 4
        self.fc4   = Module.Linear(50, 10)

    def forward(self, x):
        """Forwarding

        Args
        ----
        x : tensor-like
            Input tensor.

        Returns
        -------
        y : tensor-like
            Output tensor.

        """
        # feed forward layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # feed forward layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # flatten layer
        x = x.view(-1, 320)

        # feed forward layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # feed forward layer 4
        x = self.fc4(x)

        return x

# Hook Function Loading MNIST Dataset
def MNIST(folder, batch_size, test_batch_size):
    """Get Data Loader

    Args
    ----
    folder : Str
        Root folder of dataset.
    batch_size : Int
        Training batch size.
    test_batch_size : Int
        Validation and test batch size.

    """
    # set sample data transformation
    transform = []
    transform.append(Transform.ToTensor())
    transform.append(Transform.Normalize(mean=(0.1307,), std=(0.3081,)))
    transform = Transform.Compose(transform)

    # get data loader
    train_data = torch.utils.data.DataLoader(
        Dataset.MNIST(folder, train=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    valid_data = torch.utils.data.DataLoader(
        Dataset.MNIST(folder, train=False, transform=transform),
        batch_size=test_batch_size, shuffle=False)
    return train_data, valid_data, None

# Train the Model on Given Dataset for a Epoch
def train(data, model, criterion, optimizer, device=None):
    """Train An Epoch

    Args
    ----
    data : data-loader
        Data loader to train on.
    model : module-like
        Model to train on.
    criterion : module-like
        Loss function to use.
    optimizer : optimizer:
        Optimizer to use.
    device : Int
        CUDA device to use.

    """
    # get iterator of the dataset
    data_iter = iter(data)

    # loop all batches in the dataset
    for i in range(len(data_iter)):
        # get input and target
        input, target = next(data_iter)

        # device transfer
        if device is not None:
            input = input.cuda()
            target = target.cuda()
        else:
            pass

        # feed forward
        output = model.forward(input)
        loss = criterion.forward(output, target)

        # backpropagate
        optimizer.zero_grad()
        loss.backward()

        # update parameters
        optimizer.step()

# Averaging Variable
class AverageMeter(object):
    """Monitor Average Values"""
    def __init__(self):
        """Initialize the class"""
        self.sum = 0
        self.cnt = 0
        self.avg = None

    def update(self, val, num):
        """Update Average

        Args
        ----
        val : Float
            Averaged value to update.
        num : Int
            Number of instances of val.

        """
        self.sum += (val * num)
        self.cnt += num
        self.avg = self.sum / self.cnt

# Compute Top-k Accuracy Based on Neural Network Raw Output
def topk_accuracy(output, target, topk=(1,)):
    """Compute Top-k Accuracies

    Args
    ----
    output : tensor-like
        Output tensor telling about the probabilities of predicted labels.
    target : tensor-like
        Target tensor telling the expected labels.
    topk : Tuple of Int
        Specify the top-k settings.
        It can be several top-k values.

    Returns
    -------
    accuracy : Tuple of Float
        Accuracies corresponding to each dimension of the topk setting.

    """
    # parse arguments
    maxk = max(topk)
    dim = -1

    # get topk labels according to output
    _, label = output.topk(maxk, dim=dim, largest=True, sorted=True)

    # get the 01-matrix telling if topk labels fit with target labels
    label = label.t()
    correct = label.eq(target.view(1, -1).expand_as(label))

    # get accuracies of all dimensions of the topk setting
    num_samples = target.size(0)
    accuracy = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        accuracy.append(float(correct_k.mul_(100.0 / num_samples)))
    return tuple(accuracy)

# Evaluate the Model on Given Dataset
def evaluate(data, model, criterion, device=None):
    """Evaludate model

    Args
    ----
    data : data-loader
        Data loader to evaluate.
    model : module-like
        Model to evaluate.
    criterion : module-like
        Loss function to evaluate.
    device : Int
        CUDA device to use.

    Returns
    -------
    loss : Float
        Averaged cross entropy (loss).
    acc1 : Float
        Averaged top-1 accuracy.

    """
    # get averaging variables of loss and accuracy
    loss_avg = AverageMeter()
    acc1_avg = AverageMeter()

    # get iterator of the dataset
    data_iter = iter(data)

    # loop all batches in the dataset
    for i in range(len(data_iter)):
        # get input and target
        input, target = next(data_iter)

        # device transfer
        if device is not None:
            input = input.cuda()
            target = target.cuda()
        else:
            pass

        # feed forward
        output = model.forward(input)
        loss = criterion.forward(output, target)
        accs = topk_accuracy(output, target)
        loss_avg.update(loss.data.item(), input.size(0))
        acc1_avg.update(accs[0], input.size(0))

    return loss_avg.avg, acc1_avg.avg

# Main Process
def main(args):
    """Train and Save Parameters

    Args
    ----
    args : Dictionary
        Console arguments.

    """
    # parse arguments
    folder = args['--folder']
    bsz = args['--batch-size']
    test_bsz = args['--test-batch-size']
    num_epochs = args['--epochs']
    lr = args['--lr']
    device = args['--device']
    save = args['--save']
    resume = args['--resume']
    seed = args['--seed']

    # validate CUDA
    if device is None:
        pass
    elif torch.cuda.is_available():
        torch.cuda.set_device(device)
    else:
        print('\033[31;mCUDA is not available\033[0m')
        device = None

    # logging
    print("Training Batch Size        : {}".format(bsz))
    print("Validation/Test Batch Size : {}".format(test_bsz))
    print("Number of Epochs           : {}".format(num_epochs))
    print("Learning Rate              : {}".format(lr))
    print("CUDA GPU Device            : {}".format(device))
    print("Random Seed                : {}".format(seed))

    # get datasets
    train_data, valid_data, _ = MNIST(folder, bsz, test_bsz)

    # construct model
    model = ShallowNet()
    criterion = Module.CrossEntropyLoss()

    # resume model if necessary
    if resume is not None:
        model.load_state_dict(torch.load(resume))
    else:
        pass

    # device transfer
    if device is not None:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        pass

    # construct optimizer
    optimizer = Optimizer.SGD(model.parameters(), lr=lr)

    # evaludate on both training and validation datasets before training
    timer = time.time()
    train_loss, train_acc1 = evaluate(train_data, model, criterion, device=device)
    valid_loss, valid_acc1 = evaluate(valid_data, model, criterion, device=device)
    time_cost = int(round(time.time() - timer))

    # logging
    print("[{:>3d}/{:<3d}]"
          " (cross entropy)  Train: {:.7s}, Valid: {:.7s} |"
          " (top-1 accuracy) Train: {:.6s}%, Valid: {:.6s}% |"
          " (seconds) {}".format(
            0, num_epochs,
            str(train_loss), str(valid_loss),
            str(train_acc1), str(valid_acc1),
            time_cost))

    # train and validate for given number of epochs
    for i in range(1, num_epochs + 1):
        # train an epoch
        timer = time.time()
        train(train_data, model, criterion, optimizer, device=device)

        # evaludate on both training and validation datasets
        train_loss, train_acc1 = evaluate(train_data, model, criterion, device=device)
        valid_loss, valid_acc1 = evaluate(valid_data, model, criterion, device=device)
        time_cost = int(round(time.time() - timer))

        # logging
        print("[{:>3d}/{:<3d}]"
              " (cross entropy)  Train: {:.7s}, Valid: {:.7s} |"
              " (top-1 accuracy) Train: {:.6s}%, Valid: {:.6s}% |"
              " (seconds) {}".format(
                i, num_epochs,
                str(train_loss), str(valid_loss),
                str(train_acc1), str(valid_acc1),
                time_cost))

    # save model parameters if necessary
    if save is not None:
        torch.save(model.cpu().state_dict(), save)
    else:
        pass

if __name__ == '__main__':
    import docopt
    from schema import Schema, Use, And, Or
    args = docopt.docopt(__doc__, version='A Shallow MNIST Convolutional Net')
    requirements = {
        '--batch-size'     : And(Use(int), lambda x: x > 0,
                                 error='Training batch size should be integer > 0'),
        '--test-batch-size': And(Use(int), lambda x: x > 0,
                                 error='Test (validation) batch size should be integer > 0'),
        '--epochs'         : And(Use(int), lambda x: x > 0,
                                 error='Number of epochs should be integer > 0'),
        '--lr'             : And(Use(float), lambda x: x > 0,
                                 error='Learning rate should be float > 0'),
        '--device'         : Or(None, And(Use(int), lambda x: x > 0),
                                error='Device ID should be integer > 0'),
        '--seed'           : And(Use(int), lambda x: x > 0,
                                 error='Random seed should be integer > 0'),
        object             : object,
    }
    args = Schema(requirements).validate(args)
    main(args)
