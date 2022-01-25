import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from .grid import Grid, grid_split
import torch.backends.cudnn as cudnn

'''
    Global Constants:
        TASK_NAME: Name of the verification task (deprecated)
        PATH: The path of the model file. (Initialized in imagenet_verify)
        mean, stdvar: The normalization parameters of the data. (Initialized in imagenet_verify, default mean=(0.4914,0.4822,0.4465) stdvar=(0.2023,0.1994,0.2010))
        delta: The radius of the L-inf Ball. (Initialized in imagenet_verify, default 4/255)
        significance, error: The significance and the error rate of the PAC-Model. (Initialized in imagenet_verify, default 0.01 and 0.001)
        final_samples: The number of samples needed to calculate the final margin. (Initialized in imagenet_verify, default 1600, according to defualt error rate and significance)        
        Batchsize: The batchsize of sampling procedure. (Initialized in imagenet_verify, defualt 200)        
        device: Which device to be utilised by Pytorch. (Initialized in imagenet_verify, default 'cuda')
        model: The Pytorch Network to be verified. (Initialized in imagenet_verify)        
        pretrans: The torchvision transform to process the image. (Resize and Tensorize)
        normalization_trans: The normalization transform to normalize the data. (Initialized in imagenet_verify)
        sampling_budget: The sampling limit for each stepwise splitting. (Initialized in imagenet_verify)        
        init_grid: The Grid for Imagenet Data (224*224)
    
    Functions:
        grid_batch_sample: Grid-based Sampling for Scenario Optimization (Untargetted)
        scenario_optimization: Main Verification Function (Focused Learning, Stepwise-Splitting)
        imagenet_verify: Entry Function
'''

pretrans = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               ])
mean = (0.485, 0.456, 0.406)
stdvar = (0.229, 0.224, 0.225)
normalization_trans = transforms.Normalize(mean, stdvar)
sampling_budget = 20000
delta = 4/255
error = 1e-2
significance = 1e-3
Batchsize = 200
device = 'cuda'
init_grid = [Grid(0, 0, 224, 224)]
PATH = './models/imagenet_linf_4.pth'


def grid_batch_sample(grid_list, n_sample, batch_num, lower, upper, model, fixed_coeff=None, label=0):
    global normalization_trans, device
    feature_final = []
    result_final = []
    fixed_features = []
    # Calculate the Iteration Number
    n_iter = math.ceil(n_sample/batch_num)
    model.eval()

    for iter in range(n_iter):
        samples = np.random.uniform(lower, upper, (batch_num,)+lower.shape)
        samples_ = normalization_trans(
            torch.tensor(samples)).float().to(device)
        with torch.no_grad():
            results_ = model(samples_).cpu().detach().numpy()

        # Calculate the Untargeted Score Difference
        results_ = np.max(np.delete(results_, label, 1),
                          1) - results_[:, label]
        results_ = results_.reshape(batch_num, -1)
        result_final.append(results_)

        # Calculate the Fixed Constant
        fixed_result_i = (samples.reshape(batch_num, -1) @
                          fixed_coeff.reshape(-1)).reshape((batch_num, -1))
        fixed_features.append(fixed_result_i)

        # Calculate the Grid Sum
        feature_iter_i = []
        for grid in grid_list:
            for channel in range(3):
                grid_data = samples[:, channel, grid.leftup_x:grid.rightdown_x,
                                    grid.leftup_y:grid.rightdown_y]
                grid_sum = np.sum(grid_data, axis=1, keepdims=True)
                grid_sum = np.sum(grid_sum, axis=2, keepdims=True)
                grid_sum = grid_sum.reshape(-1, 1)
                feature_iter_i.append(grid_sum)
        # Merge the Grid Sums
        feature_iter_i = np.hstack(feature_iter_i)
        feature_final.append(feature_iter_i)

    # Merge the Batch Results
    feature_final = np.vstack(feature_final)
    result_final = np.vstack(result_final)
    fixed_features = np.vstack(fixed_features)
    return feature_final, result_final, fixed_features


def scenario_optimization(image, label):
    global significance, error, init_grid, model, sampling_budget, delta
    global pretrans, normalization_trans, Batchsize, final_samples

    # Split into 7x7 small grids (32x32 split)
    grid_list = grid_split(init_grid, 32, 32)
    img = pretrans(image)
    img_np = img.detach().numpy()
    # Calculate the Lower and Upper Bounds
    img_upper = np.clip(img_np+delta, 0., 1.)
    img_lower = np.clip(img_np-delta, 0., 1.)
    fixed_coeff = np.zeros((3, 224, 224))

    # Grid Refinement Procedure
    n_refine = 5
    for refine_step in range(n_refine):
        print('Stepwise Spliting #', refine_step, 'Start')
        print('Sampling... (%d samples)' % sampling_budget)
        features, scores, fixed_constant = grid_batch_sample(
            grid_list, sampling_budget, Batchsize, img_lower, img_upper, model, fixed_coeff, label)
        print('Constructing Template...')
        # Linear Regression to construct the Coarse Model for Stepwise Splitting
        reg = LinearRegression(fit_intercept=True).fit(
            features, scores-fixed_constant)
        coeff = np.array(reg.coef_).reshape(-1, 3)

        # Use the L2 Norm to Identify the Important Grids
        coeff_l2 = np.sqrt(np.sum(coeff*coeff, axis=1))
        coeff_l2_index = np.argsort(coeff_l2)
        coeff_l2_index_low = coeff_l2_index[:math.ceil(
            len(coeff_l2_index)*0.75)]
        coeff_l2_index_high = coeff_l2_index[math.ceil(
            len(coeff_l2_index)*0.75):]

        # Fix the Less Important Grids
        for index in coeff_l2_index_low:
            grid = grid_list[index]
            fixed_coeff[0, grid.leftup_x:grid.rightdown_x,
                        grid.leftup_y:grid.rightdown_y] = coeff[index, 0]
            fixed_coeff[1, grid.leftup_x:grid.rightdown_x,
                        grid.leftup_y:grid.rightdown_y] = coeff[index, 1]
            fixed_coeff[2, grid.leftup_x:grid.rightdown_x,
                        grid.leftup_y:grid.rightdown_y] = coeff[index, 2]

        # Split the Rest Grids
        print('Spliting...')
        grid_list_i = []
        for index in coeff_l2_index_high:
            grid_list_i.append(grid_list[index])
        grid_list = grid_split(grid_list_i, 2, 2)
        del features, scores, fixed_constant

    # Last Step, To Fix the Rest Grids
    print('Last Step...')
    features, scores, fixed_constant = grid_batch_sample(
        grid_list, sampling_budget, Batchsize, img_lower, img_upper, model, fixed_coeff, label)
    reg = LinearRegression(fit_intercept=True).fit(
        features, scores-fixed_constant)
    intercept = reg.intercept_
    coeff = np.array(reg.coef_).reshape(-1, 3)
    for index in range(len(coeff)):
        grid = grid_list[index]
        fixed_coeff[0, grid.leftup_x:grid.rightdown_x,
                    grid.leftup_y:grid.rightdown_y] = coeff[index, 0]
        fixed_coeff[1, grid.leftup_x:grid.rightdown_x,
                    grid.leftup_y:grid.rightdown_y] = coeff[index, 1]
        fixed_coeff[2, grid.leftup_x:grid.rightdown_x,
                    grid.leftup_y:grid.rightdown_y] = coeff[index, 2]
    del features, scores, fixed_constant

    # Calculate the Margin
    features, scores, fixed_constant = grid_batch_sample(
        init_grid, final_samples, Batchsize, img_lower, img_upper, model, fixed_coeff, label)
    eps_max = np.max(np.abs(scores-fixed_constant-intercept))
    print('Margin: ', eps_max)
    del features, scores, fixed_constant

    safe = True
    unsafe = False
    # Calculate the Maximum of the Learned Model, Find The Potential Counter-Example
    val_max = fixed_coeff[fixed_coeff < 0]@img_lower[fixed_coeff < 0] + \
        fixed_coeff[fixed_coeff > 0]@img_upper[fixed_coeff > 0] + \
        intercept+eps_max
    print('Evaluated Delta Max Value: ', val_max)
    if val_max > 0:
        print('Potential Counter-example Found')
        safe = False
        # Examine the Potential Counter-Example
        ce = np.zeros_like(img_lower)
        ce[fixed_coeff <= 0] = img_lower[fixed_coeff <= 0]
        ce[fixed_coeff > 0] = img_upper[fixed_coeff > 0]
        with torch.no_grad():
            ce = normalization_trans(torch.tensor(ce).unsqueeze(0)).to(device)
            scores = model(ce)[0]
            print('True Label: ', torch.argmax(scores), 'Score: ', torch.max(
                scores), 'Original Label:', label, 'Scores: ', scores[label])
            if torch.argmax(scores) != label:
                unsafe = True
                print('Conter-example Confirmed')
    if safe:
        print('Network is PAC-model robust with error rate',
              error, 'and confidence level', 1-significance)
        return 1
    elif unsafe:
        print('Unsafe. Adversarial Example Found.')
        return 0
    print('Unknown. Potential Counter-Example exists.')
    return 2


def imagenet_verify(model_class, args):
    global delta, PATH, error, significance, final_samples, normalization_trans, mean, stdvar, dataset, device, model, Batchsize, sampling_budget
    PATH = args.model
    delta = args.radius/255.
    error = args.epsilon
    significance = args.eta
    Batchsize = args.batchsize
    image_path = args.image
    final_samples = math.ceil(2/error*(math.log(1/significance)+1))
    final_samples = math.ceil(final_samples/Batchsize)*Batchsize

    model = model_class()
    model.load_state_dict(torch.load(PATH))
    if getattr(args, 'mean') != None:
        mean = args.mean
    if getattr(args, 'std') != None:
        stdvar = args.std
    if getattr(args, 'budget') != None:
        sampling_budget = args.budget
    normalization_trans = transforms.Normalize(mean, stdvar)
    if args.gpu == False:
        device = 'cpu'
    np.random.seed(0)
    if device == 'cuda':
        cudnn.deterministic = True
        cudnn.benchmark = False
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    if getattr(args, 'label') != None:
        label = args.label
    else:
        label = int(torch.argmax(model(normalization_trans(
            pretrans(image)).unsqueeze(0).to(device))[0]).cpu())
    print('True Label: ', label)

    try:
        print('Verification Radius(L-inf): ', args.radius)
        print('Mean: ', mean)
        print('Std: ', stdvar)
        return scenario_optimization(image, label)
    except Exception as err:
        print('Error: Verification Failed')
        print(err)
