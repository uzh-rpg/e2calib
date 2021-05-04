import os
import json
import time
import torch
import shutil
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from os.path import join, basename
from data.format import Events
from data.provider import DataProvider
from reconstruction.utils.timers import cuda_timers
from reconstruction.utils.loading_utils import load_model, get_device
from reconstruction.image_reconstructor import ImageReconstructor
from reconstruction.options.inference_options import set_inference_options
from reconstruction.utils.voxelgrid import VoxelGrid
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Image reconstruction')
    parser.add_argument('--h5file', help='Path to h5 file containing events for reconstruction.', default='/home/manasi/Downloads/data.h5')
    parser.add_argument('-c', '--path_to_model', type=str,
                        help='path to the model weights',
                        default='reconstruction/pretrained/E2VID_lightweight.pth.tar')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--freq_hz', '-fhz', type=int, default=30, help='Frequency for reconstructing frames from events')
    parser.add_argument('--upsample_freq', '-ufhz', type=int, default=2, help='Frequency for upsampling the frequence of recontruction')
    print_every_n = 50

    set_inference_options(parser)

    args = parser.parse_args()

    # Data loader
    h5_path = Path(args.h5file)
    freq_hz = args.freq_hz
    data_provider = DataProvider(h5_path, height=args.height, width=args.width, reconstruction_frequency_hz=args.freq_hz)


    # Load model to device
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)
    model = model.to(device)
    model.eval()

    print(args.output_folder)
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    else:
        assert os.path.isdir(args.output_folder)
    
    args.show_events = True
    image_reconstructor = ImageReconstructor(model, args.height, args.width, model.num_bins, args)
    idx =0 
    for events in data_provider:
        if events.events.size > 0:
            # reconstruction_events = events.events
            # N = 131
            # events = Events(
            #         x=np.array(np.random.randint(0, 640, (N,)), dtype='uint16'),
            #         y=np.array(np.random.randint(0, 480, (N,)), dtype='uint16'),
            #         p=np.array(np.random.randint(0, 2, (N,)), dtype='uint8'),
            #         t=np.array(np.linspace(0, 101, num= N), dtype='int64'),
            #         width=640,
            #         height=480,
            #         t_reconstruction=101)
            grid_repr = VoxelGrid(model.num_bins, events.width, events.height, upsample_rate=args.upsample_freq)
            sliced_events = grid_repr.event_slicer(events.events)
            for i in range(len(sliced_events)):
                grid, ts = grid_repr.events_to_voxel_grid(sliced_events[i])
                event_tensor= torch.from_numpy(grid)
                # print(event_tensor.shape, ts)
                image_reconstructor.update_reconstruction(event_tensor, idx, stamp=ts)
                idx+=1