import argparse
import importlib
from verify import mnist
import time
from auxilliry import Namespace

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--net', choices=['FNN1', 'FNN2', 'FNN3', 'FNN4', 'FNN5', 'FNN6', 'CNN1', 'CNN2',
                    'CNN3', 'CNN4', 'CNN5', 'CNN6'], help='Which mnist network to be verified', required=True, metavar='FNN1-6, CNN1-6')
parser.add_argument('-gpu', '--gpu', action='store_true',
                    help='Whether to use GPU (Optional, defualt False)')
parser.add_argument('-inc', '--increase', action='store_true',
                    help='Check the results of the maximum robustness radius plus 1 (Optional, defualt False)')
parser.add_argument('-solver', '--lpsolver', choices=[
                    'gurobi', 'cbc'], help='The Linear Programming Solver. (Gurobi or CBC, cvxpy default LP solver if not assigned)')
args = parser.parse_args()

ncf_dict = {'FNN1': 'models.mnist_network_FNN',
            'FNN2': 'models.mnist_network_FNN',
            'FNN3': 'models.mnist_network_FNN',
            'FNN4': 'models.mnist_network_FNN',
            'FNN5': 'models.mnist_network_FNN',
            'FNN6': 'models.mnist_network_FNN',
            'CNN1': 'models.mnist_ERAN_convs',
            'CNN2': 'models.mnist_ERAN_convs',
            'CNN3': 'models.mnist_ERAN_convs',
            'CNN4': 'models.mnist_ERAN_convs',
            'CNN5': 'models.mnist_ERAN_convs',
            'CNN6': 'models.mnist_ERAN_convs'}

nc_dict = {'FNN1': 'FNN1',
           'FNN2': 'FNN2',
           'FNN3': 'FNN3',
           'FNN4': 'FNN4',
           'FNN5': 'FNN5',
           'FNN6': 'FNN6',
           'CNN1': 'ConvSmallNetwork',
           'CNN2': 'ConvSmallNetwork',
           'CNN3': 'ConvSmallNetwork',
           'CNN4': 'ConvMedNetwork',
           'CNN5': 'ConvMedNetwork',
           'CNN6': 'ConvMedNetwork'}

model_dict = {'FNN1': './models/mnist_FNN_1.pth',
              'FNN2': './models/mnist_FNN_2.pth',
              'FNN3': './models/mnist_FNN_3.pth',
              'FNN4': './models/mnist_FNN_4.pth',
              'FNN5': './models/mnist_FNN_5.pth',
              'FNN6': './models/mnist_FNN_6.pth',
              'CNN1': './models/convSmallRELU__Point.pth',
              'CNN2': './models/convSmallRELU__PGDK.pth',
              'CNN3': './models/convSmallRELU__DiffAI.pth',
              'CNN4': './models/convMedGRELU__Point.pth',
              'CNN5': './models/convMedGRELU__PGDK_w_0.1.pth',
              'CNN6': './models/convMedGRELU__PGDK_w_0.3.pth'}

radius_dict = {'FNN1': [4, 36, 13, 11, 11, 18, 24, 36, 9, 28, 5, 12, 31, 17, 11, 16, 14, 26, 11, 11, 14, 23, 18, 11, 2],
               'FNN2': [5, 32, 14, 22, 6, 19, 26, 19, 23, 26, 25, 13, 42, 33, 21, 23, 10, 22, 11, 10, 26, 24, 6, 29, 8],
               'FNN3': [9, 40, 16, 26, 19, 22, 37, 28, 26, 33, 22, 15, 35, 25, 25, 24, 18, 10, 18, 12, 29, 29, 18, 28, 5],
               'FNN4': [11, 35, 17, 29, 5, 18, 36, 39, 27, 36, 30, 14, 47, 27, 25, 23, 23, 34, 16, 17, 23, 21, 19, 29, 15],
               'FNN5': [22, 27, 18, 41, 21, 12, 63, 23, 40, 35, 24, 18, 35, 31, 33, 25, 7, 40, 15, 23, 25, 23, 15, 49, 17],
               'FNN6': [10, 25, 17, 31, 22, 22, 35, 37, 27, 33, 26, 17, 45, 27, 21, 27, 22, 24, 18, 22, 28, 24, 15, 34, 6],
               'CNN1': [12, 59, 25, 46, 30, 27, 47, 54, 45, 48, 41, 43, 62, 48, 37, 49, 28, 32, 19, 29, 49, 38, 15, 44, 15],
               'CNN2': [28, 66, 43, 47, 47, 50, 47, 55, 49, 49, 47, 48, 61, 54, 49, 49, 49, 51, 35, 45, 47, 58, 42, 48, 33],
               'CNN3': [27, 46, 26, 34, 34, 33, 34, 43, 35, 37, 35, 36, 44, 36, 37, 36, 37, 37, 27, 33, 40, 41, 30, 34, 19],
               'CNN4': [3, 59, 30, 52, 28, 28, 50, 41, 35, 50, 37, 46, 62, 46, 41, 41, 45, 25, 21, 35, 46, 46, 18, 53, 23],
               'CNN5': [29, 60, 44, 52, 44, 50, 49, 49, 49, 52, 43, 47, 58, 53, 48, 49, 50, 49, 43, 44, 50, 56, 41, 52, 37],
               'CNN6': [27, 86, 62, 81, 67, 58, 69, 78, 52, 80, 45, 63, 86, 72, 74, 79, 73, 64, 53, 36, 77, 74, 50, 82, 28]}

mnist_args = Namespace(netclassfile=ncf_dict[args.net], netclassname=nc_dict[args.net], model=model_dict[args.net], dataset='mnist', epsilon=0.01,
                       eta=0.001, train=True, FThreshold=2000, SThreshold=8000, mean=None, std=None, label=None, image=None, batchsize=None, budget=None, lpsolver=None)
mnist_args.update(gpu=args.gpu)
mnist_args.update(lpsolver=args.lpsolver)
radius_list = radius_dict[args.net]

net_class_module = importlib.import_module(mnist_args.netclassfile)
net_class = getattr(net_class_module, mnist_args.netclassname)

start = time.time()
robust = 0
for ind in range(25):
    if args.increase:
        mnist_args.update(index=ind, radius=radius_list[ind]+1)
    else:
        mnist_args.update(index=ind, radius=radius_list[ind])
    result = mnist.mnist_verify(net_class, mnist_args)
    if result == 1:
        robust += 1
print('%d PAC-Model Robust out of 25 cases.' % robust)
print('Total Time: ', time.time()-start)
