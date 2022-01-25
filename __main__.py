import argparse
import importlib
from verify import mnist, cifar, imagenet
import time


def verify(args):
    try:
        net_class_module = importlib.import_module(args.netclassfile)
        net_class = getattr(net_class_module, args.netclassname)
    except Exception as err:
        print('Error: Import model class failed.')
        print(err)
        exit(-1)

    if args.epsilon > 1. or args.epsilon < 0.:
        print('Error: error rate should be in [0,1]')

    if args.eta > 1. or args.eta < 0.:
        print('Error: significance level should be in [0,1]')

    start = time.time()
    if args.dataset == 'mnist':
        mnist.mnist_verify(net_class, args)
    elif args.dataset == 'cifar10':
        cifar.cifar_verify(net_class, args)
    elif args.dataset == 'imagenet':
        imagenet.imagenet_verify(net_class, args)
    print('Time: ', time.time()-start)


parser = argparse.ArgumentParser()
parser.add_argument('-ncf', '--netclassfile', type=str,
                    help='Python network class file contains the network class defined by PyTorch', required=True)
parser.add_argument('-nc', '--netclassname', type=str,
                    help='Name of the network class', required=True)
parser.add_argument('-m', '--model', type=str,
                    help='Model File for the network class containing the PyTorch statedict', required=True)
parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'cifar10', 'imagenet'],
                    help='The dataset of the model can be either mnist, cifar10 or imagenet', required=True)
parser.add_argument('-r', '--radius', type=int, choices=range(0, 256),
                    help='The verification radius of the L-inf ball (0-255)', required=True, metavar='0-255')
parser.add_argument('-eps', '--epsilon', type=float,
                    help='The error rate of the PAC-model', required=True)
parser.add_argument('-eta', '--eta', type=float,
                    help='The significance level of the PAC-model (1-confidence)', required=True)
parser.add_argument('-img', '--image', type=str,
                    help='Path of the image file to be verified  (required for Imagenet models)')
parser.add_argument('-ind', '--index', type=int, default=0,
                    help='The index of the image to be verified. (required for Mnist and Cifar10 models)')
parser.add_argument('-train', '--train', action='store_true',
                    help='Set if you want to verify images in trainset. (optional, only effect on Mnist and Cifar10 models)')
parser.add_argument('-gpu', '--gpu', action='store_true',
                    help='Set to use GPU (Optional, defualt False)')
parser.add_argument('-FT', '--FThreshold', type=int, default=2000,
                    help='The sampling threshold for the first focused learning phase. (optional, only effect on Mnist and Cifar10, default 2000)')
parser.add_argument('-ST', '--SThreshold', type=int, default=8000,
                    help='The sampling threshold for the second focused learning phase. (optional, only effect on Mnist and Cifar10, default 8000)')
parser.add_argument('-b', '--budget', type=int, default=20000,
                    help='The sampling budget for stepwise splitting. (optional, only effect on Imagenet, default=20000)')
parser.add_argument('-bsize', '--batchsize', type=int, default=200,
                    help='The batchsize of the sampling procedure (optional, only effect on Imagenet and Cifar10, default=200)')
parser.add_argument('-mean', '--mean', type=tuple,
                    help='The mean used to normalize the data. (optional, (0.485, 0.456, 0.406) for Imagenet, (0.4914, 0.4822, 0.4465) for Cifar10, (0.1307,) for Mnist, by default)')
parser.add_argument('-std', '--std', type=tuple,
                    help='The standard deviation used to normalize the data. (optional, (0.229, 0.224, 0.225) for Imagenet, (0.2023, 0.1994, 0.2010) for Cifar10, (0.3081,) for Mnist, by default)')
parser.add_argument('-l', '--label', type=int, choices=range(0, 1000),
                    help='The true label of the image according to the 1000-classes Imagenet dataset. (optional, will use the output label of the neural network if not provided, only effect on Imagenet)', metavar='0-999')
parser.add_argument('-solver', '--lpsolver', choices=[
                    'gurobi', 'cbc'], help='The Linear Programming Solver. (Gurobi or CBC, cvxpy default LP solver if not assigned)')

imagenet_required = ['image']

args = parser.parse_args()
verify(args)

# print(args)
