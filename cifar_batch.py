import argparse
import importlib
from auxilliry import Namespace
import time
from verify import cifar

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--net', choices=['CNN1', 'CNN2', 'CNN3'],
                    help='Which Cifar-10 network to be verified', required=True, metavar='CNN1-3')
parser.add_argument('-gpu', '--gpu', action='store_true',
                    help='Whether to use GPU (Optional, defualt False)')
parser.add_argument('-r', '--radius', type=int, choices=[
                    2, 4, 6, 8], help='The verification radius of the L-inf ball (0-255)', required=True)
parser.add_argument('-solver', '--lpsolver', choices=[
                    'gurobi', 'cbc'], help='The Linear Programming Solver. (Gurobi or CBC, cvxpy default LP solver if not assigned)')
args = parser.parse_args()

ncf_dict = {'CNN1': 'models.cifar_ERAN_convs',
            'CNN2': 'models.cifar_ERAN_convs',
            'CNN3': 'models.cifar_ERAN_convs'}

nc_dict = {'CNN1': 'ConvSmallNetwork',
           'CNN2': 'ConvMedNetwork',
           'CNN3': 'ConvMedNetwork'}

model_dict = {'CNN1': './models/cifar_convSmallRELU__PGDK.pth',
              'CNN2': './models/cifar_convMedGRELU__PGDK_w_0.0078.pth',
              'CNN3': './models/cifar_convMedGRELU__PGDK_w_0.0313.pth'}

cifar_args = Namespace(netclassfile=ncf_dict[args.net], netclassname=nc_dict[args.net], model=model_dict[args.net], dataset='cifar10', epsilon=0.01,
                       eta=0.001, train=True, FThreshold=20000, SThreshold=10000, mean=None, std=None, label=None, image=None, batchsize=200, budget=None, lpsolver=None)
cifar_args.update(gpu=args.gpu, radius=args.radius)
cifar_args.update(lpsolver=args.lpsolver)

net_class_module = importlib.import_module(cifar_args.netclassfile)
net_class = getattr(net_class_module, cifar_args.netclassname)

robust = 0
start = time.time()
robust = 0
for ind in range(100):
    cifar_args.update(index=ind)
    result = cifar.cifar_verify(net_class, cifar_args)
    if result == 1:
        robust += 1
print('%d PAC-Model Robust out of 100 cases.' % robust)
print('Total Time: ', time.time()-start)
