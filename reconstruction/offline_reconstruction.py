import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
from utils.loading_utils import load_model, get_device
from matplotlib import pyplot as plt
from os.path import join, basename
import numpy as np
import json
import argparse
from utils.timers import cuda_timers
import time
import shutil
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
from utils.voxelgrid import VoxelGrid
from utils.data import Events

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Image reconstruction')
    parser.add_argument('-c', '--path_to_model', type=str,
                        help='path to the model weights',
                        default=os.path.join(os.environ['PRETRAINED_MODELS'], 'E2VID_lightweight.pth.tar'))
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    print_every_n = 50

    set_inference_options(parser)

    args = parser.parse_args()
    # Load model to device
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)
    model = model.to(device)
    model.eval()

    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        assert os.path.isdir(output_dir)
    
    args.output_folder = output_dir
    args.show_events = True
    image_reconstructor = ImageReconstructor(model, args.height, args.width, model.num_bins, args)
    N = 5000
    events = Events(
            x=np.array(np.random.randint(0, 640, (N,)), dtype='uint16'),
            y=np.array(np.random.randint(0, 480, (N,)), dtype='uint16'),
            p=np.array(np.random.randint(0, 2, (N,)), dtype='uint8'),
            t=np.array(np.linspace(0, 101, num= N), dtype='int64'),
            width=640,
            height=480,
            t_reconstruction=101)
    # print(events.p)
    grid_repr = VoxelGrid(5, events.width, events.height, upsample_rate=2)
    sliced_events = grid_repr.event_slicer(events)
    for i in range(len(sliced_events)):
        grid, ts = grid_repr.events_to_voxel_grid(sliced_events[i])
        event_tensor= torch.from_numpy(grid)
        print(event_tensor.shape, ts)
        image_reconstructor.update_reconstruction(event_tensor, i, stamp=ts)
        print()