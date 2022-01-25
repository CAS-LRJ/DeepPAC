import argparse
import importlib
from auxilliry import Namespace, result_dict
import time
from verify import cifar
import math

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--net', choices=['ResNet18', 'ResNet50', 'ResNet152'],
                    help='Which Cifar-10 network to be verified', required=True)
parser.add_argument('-ind', '--index', type=int,
                    choices=[0, 1, 2, 3, 4], help='The index of the image to be verified.')
parser.add_argument('-gpu', '--gpu', action='store_true',
                    help='Whether to use GPU (Optional, defualt False)')
parser.add_argument('-solver', '--lpsolver', choices=[
                    'gurobi', 'cbc'], help='The Linear Programming Solver. (Gurobi or CBC, cvxpy default LP solver if not assigned)')
args = parser.parse_args()

ncf_dict = {'ResNet18': 'models.resnet',
            'ResNet50': 'models.resnet',
            'ResNet152': 'models.resnet'}

nc_dict = {'ResNet18': 'ResNet18',
           'ResNet50': 'ResNet50',
           'ResNet152': 'ResNet152'}

model_dict = {'ResNet18': './models/Cifar_ResNet18_adv_trim.pth',
              'ResNet50': './models/Cifar_ResNet50_adv_trim.pth',
              'ResNet152': './models/Cifar_ResNet152_adv_trim.pth'}

radius_dict = {'ResNet18_0': [5, 4, 5, 4, 5, 4],
               'ResNet50_0': [8, 8, 8, 8, 9, 8],
               'ResNet152_0': [5, 5, 5, 5, 5, 5],
               'ResNet18_1': [16, 14, 15, 14, 15, 14],
               'ResNet50_1': [12, 11, 12, 12, 12, 11],
               'ResNet152_1': [10, 9, 10, 9, 10, 9],
               'ResNet18_2': [11, 10, 11, 10, 11, 10],
               'ResNet50_2': [6, 5, 6, 5, 6, 5],
               'ResNet152_2': [9, 8, 9, 8, 9, 8],
               'ResNet18_3': [1, 1, 1, 1, 1, 1],
               'ResNet50_3': [3, 3, 3, 3, 3, 3],
               'ResNet152_3': [6, 5, 6, 5, 6, 5],
               'ResNet18_4': [16, 13, 16, 14, 16, 14],
               'ResNet50_4': [17, 15, 17, 15, 17, 15],
               'ResNet152_4': [12, 10, 12, 10, 12, 10]}

cifar_args = Namespace(netclassfile=ncf_dict[args.net], netclassname=nc_dict[args.net], model=model_dict[args.net], dataset='cifar10', epsilon=0.01,
                       eta=0.001, train=True, FThreshold=20000, SThreshold=10000, mean=None, std=None, label=None, image=None, batchsize=200, budget=None, lpsolver=None)
cifar_args.update(gpu=args.gpu, index=args.index)
cifar_args.update(lpsolver=args.lpsolver)

Free_v = 50

net_class_module = importlib.import_module(cifar_args.netclassfile)
net_class = getattr(net_class_module, cifar_args.netclassname)

radius_list = radius_dict[args.net+'_'+str(args.index)]
i = 0
results = ''

start = time.time()
for epsilon, eta in [(0.01, 0.001), (0.1, 0.001), (0.01, 0.1)]:
    for FThreshold in [20000, 5000]:
        radius = radius_list[i]
        i = i+1
        SThreshold = math.ceil(2/epsilon*(math.log(1/eta)+1+Free_v))

        cifar_args.update(epsilon=epsilon, eta=eta,
                          FThreshold=FThreshold, SThreshold=SThreshold, radius=radius)
        result = cifar.cifar_verify(net_class, cifar_args)
        results += 'Epsilon: %g Eta: %g K(1): %d Radius(L-inf): %d    ' % (
            epsilon, eta, FThreshold, radius)
        results += result_dict[result] + '\n'

        cifar_args.update(epsilon=epsilon, eta=eta, FThreshold=FThreshold,
                          SThreshold=SThreshold, radius=radius+1)
        result = cifar.cifar_verify(net_class, cifar_args)
        results += 'Epsilon: %g Eta: %g K(1): %d Radius(L-inf): %d    ' % (
            epsilon, eta, FThreshold, radius+1)
        results += result_dict[result] + '\n'

print('Time: ', time.time()-start)
print(results)
