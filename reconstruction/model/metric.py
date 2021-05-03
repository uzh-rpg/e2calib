'''
This code is provided for internal research and development purposes by Huawei solely,
in accordance with the terms and conditions of the research collaboration agreement of May 7, 2020.
Any further use for commercial purposes is subject to a written agreement.
'''
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import torch
from LPIPS.models import dist_model as dm
import numpy as np
from scipy import ndimage


def mse(y_input, y_target):
    N, C, H, W = y_input.shape
    assert(C == 1 or C == 3)
    sum_mse_over_batch = 0.

    for i in range(N):
        sum_mse_over_batch += mean_squared_error(
            y_input[i, 0, :, :], y_target[i, 0, :, :])

        if C == 3:  # color
            sum_mse_over_batch += mean_squared_error(
                y_input[i, 1, :, :], y_target[i, 1, :, :])
            sum_mse_over_batch += mean_squared_error(
                y_input[i, 2, :, :], y_target[i, 2, :, :])

    mean_mse = sum_mse_over_batch / (float(N))
    if C == 3:
        mean_mse /= 3.0

    return mean_mse


def structural_similarity(y_input, y_target):
    N, C, H, W = y_input.shape
    assert(C == 1 or C == 3)
    # N x C x H x W -> N x W x H x C -> N x H x W x C
    y_input = np.swapaxes(y_input, 1, 3)
    y_input = np.swapaxes(y_input, 1, 2)
    y_target = np.swapaxes(y_target, 1, 3)
    y_target = np.swapaxes(y_target, 1, 2)
    sum_structural_similarity_over_batch = 0.
    for i in range(N):
        if C == 3:
            sum_structural_similarity_over_batch += ssim(
                y_input[i, :, :, :], y_target[i, :, :, :], multichannel=True)
        else:
            sum_structural_similarity_over_batch += ssim(
                y_input[i, :, :, 0], y_target[i, :, :, 0])

    return sum_structural_similarity_over_batch / float(N)


""" Perceptual distance """
use_gpu = True
device = torch.device('cuda:0') if use_gpu else torch.device('cpu')
# Initializing the model
model = dm.DistModel()
# Linearly calibrated models
model.initialize(model='net-lin', net='vgg', use_gpu=use_gpu, spatial=False)
print('Perceptual Distance: model [%s] initialized' % model.name())


def perceptual_distance(y_input, y_target):

    # input are numpy arrays of size (N, C, H, W) in the range [0,1]
    # for the perceptual distance, we need Tensors of size (N,3,128,128) in the range [-1,1]
    y_input_torch = torch.from_numpy(y_input).to(device)
    y_target_torch = torch.from_numpy(y_target).to(device)

    if y_input_torch.shape[1] == 1:
        y_input_torch = torch.cat(
            [y_input_torch, y_input_torch, y_input_torch], dim=1)
        y_target_torch = torch.cat(
            [y_target_torch, y_target_torch, y_target_torch], dim=1)

    # normalize to [-1,1]
    y_input_torch = 2 * y_input_torch - 1
    y_target_torch = 2 * y_target_torch - 1

    dist = model.forward(y_input_torch, y_target_torch)

    return dist.mean()
