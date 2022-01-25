import cvxpy as cp
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

'''
    Global Constants:
        TASK_NAME: Name of the verification task (deprecated)
        PATH: The path of the model file. (Initialized in mnist_verify)
        mean, stdvar: The normalization parameters of the data. (Initialized in mnist_verify, default mean=(0.4914, 0.4822, 0.4465) stdvar=(0.2023, 0.1994, 0.2010))
        image_index: The index of the image to be verified. (Initialized in mnist_verify, default 256)
        delta: The radius of the L-inf Ball. (Initialized in mnist_verify, default 2/255)
        significance, error: The significance and the error rate of the PAC-Model. (Initialized in mnist_verify, default 0.01 and 0.001)
        final_samples: The number of samples needed to calculate the final margin. (Initialized in mnist_verify, default 1600, according to defualt error rate and significance)
        N, Output_size: The input size and output size of MNIST data. (784 and 10)
        var_l, var_u: The bounds of parameters to be learned by scenario optimization. (-100 and 100)
        FThreshold, SThreshold: The sampling limits for the first phase and second phase of focused learning. (Initialized in mnist_verify, defualt 20000 and 10000)
        dataset: Which MNIST dataset to be verified. (Initialized in mnist_verify, defualt trainset)
        device: Which device to be utilised by Pytorch. (Initialized in mnist_verify, default 'cuda')
        net: The Pytorch Network to be verified. (Initialized in Prepare)
        image, label: The image and label get from dataset according to the index. (Initialized in Prepare)
        image_l, image_u: The upper bounds and lower bounds of the L-inf ball. (Initialized in Prepare)
        images_first, output_first: The samples and outputs for first phase focused learning. (Initialized in Prepare)
        images_second, output_second: The samples and outputs for second phase focused learning. (Initialized in Prepare)
        images_third, output_third: The samples and outputs for final margin calculation. (Initialized in Prepare)
        normalization_trans: The normalization transform to normalize the data. (Initialized in mnist_verify)
        lp_solver: The Linear Programming Solver (Initialized in mnist_verify)
    
    Functions:
        Prepare: Preparation for Scenario Optimization (Model and Data)
        scenario_optimization: Main Verification Function (Component Learning, Focused Learning)
        mnist_verify: Entry Function
'''

TASK_NAME = 'example'
mean = (0.1307,)
stdvar = (0.3081,)
# Correctness: 1-error
error = 0.01
# Confidence: 1-significance
significance = 1e-3
# Final Samples
final_samples = math.ceil(2/error*(math.log(1/significance)+1))
N = 784
Output_size = 10
var_l = -100
var_u = 100
# First Try Threshold
FThreshold = 2000
# Second Try Threshold
SThreshold = 8000
# Fix the random seed
np.random.seed(0)
# trainset by default
dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                     download=True, transform=transforms.Compose([transforms.ToTensor()]))
# Use GPU by default
device = 'cuda'


def add_delta(x):
    global delta
    return 1. if x+delta > 1. else x+delta


def dec_delta(x):
    global delta
    return 0. if x-delta < 0. else x-delta


def Prepare(model):
    global dataset, net, normalization_trans, image, label, image_l, image_u, images_first, images_second, images_third, output_first, output_second, output_third, delta, device
    # Fix the random seed
    np.random.seed(0)

    # Move Network to device, turn on cuda deterministic
    if device == 'cuda':
        cudnn.benchmark = False
        cudnn.deterministic = True
    net = model().to(device)
    net.load_state_dict(torch.load(PATH, map_location=device))
    net.eval()

    # Get Image and Label
    image, label = dataset[image_index]
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
        torch.from_numpy(image_list_first).float()).to(device)

    # Second Sampling Proceudre
    image_list_second = []
    for i in range(SThreshold):
        image_list_second.append(np.random.uniform(image_l, image_u))
    image_list_second = np.array(image_list_second)
    images_second = normalization_trans(
        torch.from_numpy(image_list_second).float()).to(device)

    # Third Sampling Proceudre
    image_list_third = []
    for i in range(final_samples):
        image_list_third.append(np.random.uniform(image_l, image_u))
    image_list_third = np.array(image_list_third)
    images_third = normalization_trans(
        torch.from_numpy(image_list_third).float()).to(device)

    # Prepare the output data
    output_first = None
    output_second = None
    output_origin = None
    output_third = None
    with torch.no_grad():
        # Get the Original Output
        output_origin = net(normalization_trans(
            image.clone().detach().unsqueeze(0).float()).to(device))
        print('Original Output:', output_origin)

        # Forward the samples
        output_first = net(images_first)
        output_second = net(images_second)
        output_third = net(images_third)

    # Save the samples and outputs in RAM
    images_first = images_first.cpu().detach()
    images_second = images_second.cpu().detach()
    images_third = images_third.cpu().detach()
    output_first = output_first.cpu().detach()
    output_second = output_second.cpu().detach()
    output_third = output_third.cpu().detach()


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
            print('Current Optimization:', output_node)
            # First Phase Focused Learning
            print('Constructing Template (First Phase Focused Learning) with',
                  FThreshold, 'Cases')
            net_out = output_first[:, output_node].numpy(
            )-output_first[:, label].numpy()
            left = np.append(images_first.reshape(
                FThreshold, -1), np.ones((FThreshold, 1)), axis=1)
            cons = [var >= var_l, var <= var_u, left@var -
                    net_out <= eps, net_out-left@var <= eps, eps >= 0]
            obj = cp.Minimize(eps)
            prob = cp.Problem(obj, cons)
            prob.solve(solver=lp_solver, warm_start=True)
            var_save = np.array(var.value)

            # Second Phase Focused Learning
            print('Scenario Optimization (Second Phase Focused Learning) with',
                  SThreshold, 'Cases')
            # Prepare Free Variables
            # choose Free Variables with largest score
            score = np.abs(np.array(var.value))
            score_i = score.argsort()[-Free_v:]
            score_i.sort()
            var_f = var[score_i]
            constant_v = []
            # Fix Other Variables
            for i in range(K):
                if i in score_i:
                    constant_v.append(0)
                else:
                    constant_v.append(var[i].value)
            constant_v = np.array(constant_v).reshape(K,)

            # Solve the LP
            net_out = output_second[:, output_node].numpy(
            )-output_second[:, label].numpy()
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
            eps_final_list.append(np.max(np.abs(
                left@var_save-(output_third[:, output_node].numpy()-output_third[:, label].numpy()))))
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
                candidate.append(np.array(pce).reshape(1, 28, 28))
                candidate_i.append(verification_node)

    # If No Potential Adversarial Example, PAC-Model Robust
    if len(candidate) == 0:
        print('Network is PAC-model robust with error rate',
              error, 'and confidence level', 1-significance)
        return 1

    # Potential Conter-Example Examination
    flag = False
    for example_i, example in zip(candidate_i, candidate):
        ce_tensor = normalization_trans(
            torch.from_numpy(example).unsqueeze(0).float())
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
                print('Differential:', ce_output[example_i]-ce_output[label])
            else:
                print('Fake Counter-Example. Potential Attack label:', example_i, 'Score:',
                      ce_output[example_i], 'Real Largest Label:', ce_label, 'Score:', ce_output[ce_label], 'Original Label Score:', ce_output[label])
                print('Differential:', ce_output[example_i]-ce_output[label])

    if flag:
        print('Unsafe. Adversarial Example Found.')
        return 0
    else:
        print('Unknown. Potential Counter-Example exists.')
        return 2


def mnist_verify(model_class, args):
    global delta, PATH, image_index, FThreshold, SThreshold, error, significance, final_samples, normalization_trans, mean, stdvar, dataset, device, lp_solver
    PATH = args.model
    delta = args.radius/255.
    error = args.epsilon
    image_index = args.index
    significance = args.eta
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
    if args.train == False:
        dataset = torchvision.datasets.MNIST(root='./data', train=False,
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
