r"""
    Configure

    (0) io,
    (1) hardware,
    (2) seed, 
    (3) nets, 
    (4) acts, 
    (5) preset, 
    (6) optimisation, 
    (7) modes, 
    (8) tunning,
    (9) paths
"""
import argparse

from torch import device, manual_seed, flatten, no_grad, save
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.cuda import is_available as GPUcard
from torch.backends.mps import is_available as MACcard

from torch.nn import Conv2d, Dropout, Linear, Softmax

from torch.nn.functional import relu, max_pool2d, leaky_relu, \
    tanh, tanhshrink, sigmoid, log_softmax, \
        nll_loss, cross_entropy, mse_loss

from torch.optim import Adadelta, Adagrad, Adamax, RMSprop
from torch.optim.lr_scheduler import StepLR, MultiStepLR

class io:
    def parser():
        parser = argparse.ArgumentParser(
            description='PyTorch MNIST Example')
        parser.add_argument(
            '--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
        parser.add_argument(
            '--test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
        parser.add_argument(
            '--epochs', type=int, default=14, metavar='N',
            help='number of epochs to train (default: 14)')
        parser.add_argument(
            '--lr', type=float, default=1.0, metavar='LR',
            help='learning rate (default: 1.0)')
        parser.add_argument(
            '--gamma', type=float, default=0.7, metavar='M',
            help='Learning rate step gamma (default: 0.7)')
        parser.add_argument(
            '--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
        parser.add_argument(
            '--no-mps', action='store_true', default=False,
            help='disables macOS GPU training')
        parser.add_argument(
            '--dry-run', action='store_true', default=False,
            help='quickly check a single pass')
        parser.add_argument(
            '--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
        parser.add_argument(
            '--log-interval', type=int, default=10, metavar='N',
            help='how many batches to wait before logging train status')
        parser.add_argument(
            '--save-model', action='store_true', default=False,
            help='for saving the current model')
        return parser.parse_args()
    def writter(save_dir, save_name):
        return save(save_dir, save_name)
    def data1(data_dir, train, download, transform):
        return datasets.MNIST(root=data_dir, train=train, 
                              transform=transform, download=download)
    def data2(data_dir, train, download, transform):
        return datasets.CIFAR10(root=data_dir, train=train, 
                                transform=transform, download=download)
    def data3(data_dir, train, download, transform):
        return datasets.ImageNet(root=data_dir, train=train, 
                                 transform=transform, download=download)
    def dataloader(dataset,**kwargs):
        return DataLoader(dataset,**kwargs)
    cleaner = transforms

class hardware:
    graphical1 = GPUcard
    graphical2 = MACcard
    def deploy(choose_gpu,choose_mps):
        if choose_gpu:
            return device("cuda")
        elif choose_mps:
            return device("mps")
        else:
            return device("cpu")

class seed:
    def seed1(seed):
        return manual_seed(seed)

class nets:
    net1 = Conv2d
    net2 = Dropout
    net3 = Linear
    net4 = Softmax

class acts:
    act0 = flatten
    act1 = relu
    act2 = max_pool2d
    act3 = leaky_relu
    act4 = tanh
    act5 = tanhshrink
    act6 = sigmoid
    act99= log_softmax

class preset:
    n_input_conv1, n_output_conv1 = 1, 32
    n_input_conv2, n_output_conv2 = 32, 64
    n_kernel_conv1, n_stride_conv1 = 3, 1
    n_kernel_conv2, n_stride_conv2 = 3, 1
    p_dropout_1 = 0.25
    p_dropout_2 = 0.50
    n_input_fc1, n_output_fc1 = 9216, 128
    n_input_fc2, n_output_fc2 = 128, 10

class optimisation:
    loss1 = nll_loss
    loss2 = cross_entropy
    loss3 = mse_loss
    solver1 = Adadelta
    solver2 = Adagrad
    solver3 = Adamax
    solver4 = RMSprop
    scheduler1 = StepLR
    scheduler2 = MultiStepLR

class modes:
    def silent():
        return no_grad()

class tunnings:
    lr = 1

class paths:
    data_dir = 'data'
    checkpoints_dir = 'results/checkpoints'
    model_dir = 'models/pt'