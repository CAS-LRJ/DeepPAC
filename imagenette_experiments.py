import argparse
import importlib
from auxilliry import Namespace
import time
from verify import imagenette

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--net', choices=['ResNet50a', 'ResNet50b'],
                    help='Which imagenet network to be verified', required=True)
parser.add_argument('-gpu', '--gpu', action='store_true',
                    help='Whether to use GPU (Optional, defualt False)')
args = parser.parse_args()

ncf_dict = {'ResNet50a': 'torchvision.models',
            'ResNet50b': 'torchvision.models'}

nc_dict = {'ResNet50a': 'resnet50',
           'ResNet50b': 'resnet50'}

model_dict = {'ResNet50a': './models/imagenet_linf_4.pth',
              'ResNet50b': './models/imagenet_linf_8.pth'}

imagenette_args = Namespace(netclassfile=ncf_dict[args.net], netclassname=nc_dict[args.net], model=model_dict[args.net], dataset='imagenette', epsilon=0.01,
                            eta=0.001, train=True, FThreshold=None, SThreshold=None, mean=None, std=None, label=None, image=None, batchsize=200, budget=20000, index=None, lpsolver=None)
imagenette_args.update(gpu=args.gpu, radius=4)


net_class_module = importlib.import_module(imagenette_args.netclassfile)
net_class = getattr(net_class_module, imagenette_args.netclassname)

start = time.time()
total_safe = 0
total_unsafe = 0
tag_list = ['tench', 'english_springer', 'cassette_player', 'chain_saw',
            'church', 'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute']
for label, tag in enumerate(tag_list):
    for i in range(2):
        imagenette_args.update(
            image='./ImageNet_Samples/'+tag+'_'+str(i)+'.JPEG')
        result = imagenette.imagenette_verify(net_class, imagenette_args)
        if result == 1:
            total_safe += 1
        elif result == 0:
            total_unsafe += 1
print('Total:', 20, 'Verified :', total_safe)
print('Total Time:', time.time()-start)
