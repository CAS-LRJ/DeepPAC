import cvxpy as cp
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from sklearn.linear_model import LinearRegression

'''
    Global Constants:
        TASK_NAME: Name of the verification task (deprecated)
        PATH: The path of the model file. (Initialized in cifar_verify)
        mean, stdvar: The normalization parameters of the data. (Initialized in cifar_verify, default mean=(0.4914,0.4822,0.4465) stdvar=(0.2023,0.1994,0.2010))
        image_index: The index of the image to be verified. (Initialized in cifar_verify, default 256)
        delta: The radius of the L-inf Ball. (Initialized in cifar_verify, default 2/255)
        significance, error: The significance and the error rate of the PAC-Model. (Initialized in cifar_verify, default 0.01 and 0.001)
        final_samples: The number of samples needed to calculate the final margin. (Initialized in cifar_verify, default 1600, according to defualt error rate and significance)
        N, Output_size: The input size and output size of CIFAR-10 data. (3072 and 10)
        var_l, var_u: The bounds of parameters to be learned by scenario optimization. (-100 and 100)
        Batchsize: The batchsize of sampling procedure. (Initialized in cifar_verify, defualt 200)
        FThreshold, SThreshold: The sampling limits for the first phase and second phase of focused learning. (Initialized in cifar_verify, defualt 20000 and 10000)
        dataset: Which cifar-10 dataset to be verified. (Initialized in cifar_verify, defualt trainset)
        device: Which device to be utilised by Pytorch. (Initialized in cifar_verify, default 'cuda')
        net: The Pytorch Network to be verified. (Initialized in Prepare)
        image_l, image_u: The upper bounds and lower bounds of the L-inf ball. (Initialized in Prepare)
        images_first, output_first: The samples and outputs for first phase focused learning. (Initialized in Prepare)
        images_second, output_second: The samples and outputs for second phase focused learning. (Initialized in Prepare)
        images_third, output_third: The samples and outputs for final margin calculation. (Initialized in Prepare)
        normalization_trans: The normalization transform to normalize the data. (Initialized in cifar_verify)
        lp_solver: The Linear Programming Solver (Initialized in cifar_verify)
    
    Functions:
        Prepare: Preparation for Scenario Optimization (Model and Data)
        scenario_optimization: Main Verification Function (Component Learning, Focused Learning)
        cifar_verify: Entry Function
'''

TASK_NAME = 'example'
mean = (0.4914, 0.4822, 0.4465)
stdvar = (0.2023, 0.1994, 0.2010)
image_index = 256
delta = 2/255
# Correctness: 1-error
error = 0.01
# Confidence: 1-significance
significance = 1e-3
# Final Samples
final_samples = 1600
N = 3072
Output_size = 10
var_l = -100
var_u = 100
# Threshold Batchsize
Batchsize = 200
# First Try Threshold (multiple of batchsize)
FThreshold = 20000
# Second Try Threshold (multiple of batchsize)
SThreshold = 10000
# Fix the random seed
np.random.seed(0)
# trainset by default
dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transforms.Compose([transforms.ToTensor()]))
# Use GPU by default
device = 'cuda'
# Fix the random seed
np.random.seed(0)


def add_delta(x):
    global delta
    return 1. if x+delta > 1. else x+delta


def dec_delta(x):
    global delta
    return 0. if x-delta < 0. else x-delta


def Prepare(Net):
    global dataset, net, normalization_trans, image, label, image_l, image_u, images_first, images_second, images_third, output_first, output_second, output_third, delta, device, image_index, final_samples
    # Fix the random seed
    np.random.seed(0)

    # Move Network to device, turn on cuda deterministic
    net = Net().to(device)
    if device == 'cuda':
        cudnn.benchmark = False
        cudnn.deterministic = True
    checkpoint = torch.load(PATH, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()

    # Get Image and Label
    image, label = dataset[image_index]
    image = image.numpy()
    print('Label to verify:', label)
    add_delta_v = np.vectorize(add_delta)
    dec_delta_v = np.vectorize(dec_delta)

    # Get the upper and lower bounds
    image_l = dec_delta_v(image)
    image_u = add_delta_v(image)

    # First Sampling Proceudre
    image_list_first = []
    for i in range(FThreshold):
        image_list_first.append(np.random.uniform(image_l, image_u))
    image_list_first = np.array(image_list_first)
    images_first = normalization_trans(
        torch.from_numpy(image_list_first).float())

    # Second Sampling Proceudre
    image_list_second = []
    for i in range(SThreshold):
        image_list_second.append(np.random.uniform(image_l, image_u))
    image_list_second = np.array(image_list_second)
    images_second = normalization_trans(
        torch.from_numpy(image_list_second).float())

    # Third Sampling Proceudre
    image_list_third = []
    for i in range(final_samples):
        image_list_third.append(np.random.uniform(image_l, image_u))
    image_list_third = np.array(image_list_third)
    images_third = normalization_trans(
        torch.from_numpy(image_list_third).float())

    # Prepare the output data
    output_first = None
    output_second = None
    output_origin = None
    output_third = None
    with torch.no_grad():

        # Get the Original Output
        input_gpu = normalization_trans(torch.tensor(
            image).unsqueeze(0).float()).to(device)
        output_gpu = net(input_gpu)
        output_origin = output_gpu.cpu()
        del input_gpu
        del output_gpu
        print('Original Output:', output_origin)

        # Batch Forward the first sample
        output_first = None
        for batch_i in range(int(FThreshold/Batchsize)):
            input_gpu = images_first[batch_i *
                                     Batchsize:(batch_i+1)*Batchsize].to(device)
            output_gpu = net(input_gpu)
            if output_first is None:
                output_first = output_gpu.cpu().detach().numpy()
            else:
                output_first = np.append(
                    output_first, output_gpu.cpu(), axis=0)
            del input_gpu
            del output_gpu

        # Batch Forward the second sample
        output_second = None
        for batch_i in range(int(SThreshold/Batchsize)):
            input_gpu = images_second[batch_i *
                                      Batchsize:(batch_i+1)*Batchsize].to(device)
            output_gpu = net(input_gpu)
            if output_second is None:
                output_second = output_gpu.cpu().detach().numpy()
            else:
                output_second = np.append(
                    output_second, output_gpu.cpu(), axis=0)
            del input_gpu
            del output_gpu

        # Batch Forward the third sample
        output_third = None
        for batch_i in range(int(final_samples/Batchsize)):
            input_gpu = images_third[batch_i *
                                     Batchsize:(batch_i+1)*Batchsize].to(device)
            output_gpu = net(input_gpu)
            if output_third is None:
                output_third = output_gpu.cpu().detach().numpy()
            else:
                output_third = np.append(
                    output_third, output_gpu.cpu(), axis=0)
            del input_gpu
            del output_gpu


def scenario_optimization():
    global net, normalization_trans
    global significance, error, N, Output_size
    global image, label, image_l, image_u, images_first, images_second, images_third, output_first, output_second, output_third
    global FThreshold, SThreshold, final_samples, lp_solver

    # Linear Template
    K = N+1

    # Prepare the optimization variable
    var = cp.Variable(K)
    eps = cp.Variable(1)

    # Check the Threshold
    print('Error Ratio:', error)
    Free_v = min(math.floor(SThreshold*error/2-math.log(1/significance)-1), K)
    if Free_v < 0:
        raise Exception(
            'The Second Threshold for Scenario Optimization is too Small!')
    # The Free Varialbles in Second Phase
    print('Free Variables:', Free_v)

    # Component-wise Learning
    ans_list = []
    eps_list = []
    eps_final_list = []
    for output_node in range(Output_size):
        if output_node != label:
            print('Currently Optimization:', output_node)
            # First Phase Focused Learning
            print('Constructing Template (First Phase Focused Learning, Regression) with',
                  FThreshold, 'Cases')
            net_out = output_first[:, output_node]-output_first[:, label]
            left = np.append(images_first.reshape(
                FThreshold, -1), np.ones((FThreshold, 1)), axis=1)
            reg = LinearRegression(fit_intercept=False).fit(left, net_out)
            var_save = reg.coef_

            # Second Phase Focused Learning
            print('Scenario Optimization (Second Phase Focused Learning) with',
                  SThreshold, 'Cases')
            # Prepare Free Variables
            # choose Free Variables with largest score
            score = np.abs(var_save)
            score_i = score.argsort()[-Free_v:]
            score_i.sort()
            var_f = var[score_i]
            constant_v = []
            for i in range(K):
                if i in score_i:
                    constant_v.append(0)
                else:
                    constant_v.append(var_save[i])
            constant_v = np.array(constant_v).reshape(K,)

            # Solve the LP
            net_out = output_second[:, output_node]-output_second[:, label]
            left = np.append(images_second.reshape(
                SThreshold, -1), np.ones((SThreshold, 1)), axis=1)

            # Calculate The Constant of Fixed Variables
            constant_v = (left@constant_v)

            left = left[:, score_i]
            cons = [var_f >= var_l, var_f <= var_u, left@var_f+constant_v -
                    net_out <= eps, net_out-left@var_f-constant_v <= eps, eps >= 0]
            obj = cp.Minimize(eps)
            prob = cp.Problem(obj, cons)
            prob.solve(solver=lp_solver, warm_start=True)
            # Fix the Rest Variables
            var_save[score_i] = np.array(var_f.value)
            ans_list.append(var_save)
            eps_list.append(eps.value)
            left = np.append(images_third.reshape(
                final_samples, -1), np.ones((final_samples, 1)), axis=1)
            # Calculate the Component Margin
            eps_final_list.append(np.max(
                np.abs(left@var_save-(output_third[:, output_node]-output_third[:, label]))))
        else:
            ans_list.append(None)
            eps_list.append(0)
            eps_final_list.append(0)

    # Verification
    print('Verifying...')
    image_l_norm = (normalization_trans(torch.from_numpy(
        image_l).float().unsqueeze(0))).numpy().reshape(-1)
    image_u_norm = (normalization_trans(torch.from_numpy(
        image_u).float().unsqueeze(0))).numpy().reshape(-1)
    image_l1 = image_l.reshape(-1)
    image_u1 = image_u.reshape(-1)
    candidate = []
    candidate_i = []
    eps_list = [float(item) for item in eps_list]
    print('Component Margins:', eps_list)
    eps_max = max(eps_final_list)
    print('Margin:', eps_max)
    for verification_node in range(Output_size):
        if verification_node != label:
            f_differ = ans_list[verification_node]
            largest_v = 0
            pce = []
            # Calculate the Maximum of the Learned Model
            for i in range(K):
                if i == K-1:
                    # Constant
                    largest_v += f_differ[i]
                else:
                    if f_differ[i] > 0:
                        largest_v += f_differ[i]*image_u_norm[i]
                        pce.append(image_u1[i])
                    else:
                        largest_v += f_differ[i]*image_l_norm[i]
                        pce.append(image_l1[i])
            largest_v += eps_max

            # Check if We Find a Potential Counter-Example
            if largest_v > 0:
                print('Potential Counter-Example Found! Attack From',
                      verification_node)
                print('Difference Evaluation(', verification_node, '-', label,
                      '): [', largest_v-eps_list[verification_node]*2, ',', largest_v, ']')
                candidate.append(np.array(pce).reshape(3, 32, 32))
                candidate_i.append(verification_node)

    # If No Potential Adversarial Example, PAC-Model Robust
    if len(candidate) == 0:
        print('Network is PAC-model robust with error rate',
              error, 'and confidence level', 1-significance)
        return 1

    # Potential Conter-Example Examination
    flag = False
    for example_i, example in zip(candidate_i, candidate):
        ce_tensor = normalization_trans(torch.from_numpy(
            example).unsqueeze(0).float()).to('cuda')
        with torch.no_grad():
            ce_output = net(ce_tensor)
            _, ce_label = torch.max(ce_output.data, 1)
            ce_output = ce_output.squeeze()
            ce_label = int(ce_label)

            # Check if it is True Counter-Example
            if ce_label != label:
                flag = True
                print('Counter-Example Found! Potential Attack label:', example_i, 'Score:',
                      ce_output[example_i], 'Real Largest Label:', ce_label, 'Score:', ce_output[ce_label], 'Original Label Score:', ce_output[label])
                print('Difference:', ce_output[example_i]-ce_output[label])
            else:
                print('Fake Counter-Example. Potential Attack label:', example_i, 'Score:',
                      ce_output[example_i], 'Real Largest Label:', ce_label, 'Score:', ce_output[ce_label], 'Original Label Score:', ce_output[label])
                print('Difference:', ce_output[example_i]-ce_output[label])
            del ce_output
            del ce_tensor

    if flag:
        print('Unsafe. Adversarial Example Found.')
        return 0
    else:
        print('Unknown. Potential Counter-Example exists.')
        return 2


def cifar_verify(model_class, args):
    global delta, PATH, image_index, FThreshold, SThreshold, error, significance, final_samples, normalization_trans, mean, stdvar, dataset, device, Batchsize, lp_solver
    PATH = args.model
    delta = args.radius/255.
    error = args.epsilon
    image_index = args.index
    significance = args.eta
    Batchsize = args.batchsize
    if args.lpsolver == 'gurobi':
        lp_solver = cp.GUROBI
    elif args.lpsolver == 'cbc':
        lp_solver = cp.CBC
    else:
        lp_solver = None
    final_samples = math.ceil(2/error*(math.log(1/significance)+1))
    if getattr(args, 'mean') != None:
        mean = args.mean
    if getattr(args, 'std') != None:
        stdvar = args.std
    normalization_trans = transforms.Normalize(mean, stdvar)
    if getattr(args, 'FThreshold') != None:
        FThreshold = args.FThreshold
    if getattr(args, 'SThreshold') != None:
        SThreshold = args.SThreshold
    final_samples = math.ceil(final_samples/Batchsize)*Batchsize
    FThreshold = math.ceil(FThreshold/Batchsize)*Batchsize
    SThreshold = math.ceil(SThreshold/Batchsize)*Batchsize
    if args.train == False:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transforms.Compose([transforms.ToTensor()]))
    if args.gpu == False:
        device = 'cpu'
    try:
        print('Verification Radius(L-inf): ', args.radius)
        print('Mean: ', mean)
        print('Std: ', stdvar)
        Prepare(model_class)
        return scenario_optimization()
    except Exception as err:
        print('Error: Verification Failed')
        print(err)
