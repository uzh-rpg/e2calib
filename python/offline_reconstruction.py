import argparse
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
from pathlib import Path
import urllib

import torch
import tqdm

from data.provider import DataProvider
from data.rectimestamps import TimestampProviderFile, TimestampProviderRate
from reconstruction.utils.loading_utils import load_model, get_device
from reconstruction.image_reconstructor import ImageReconstructor
from reconstruction.options.inference_options import set_inference_options
from reconstruction.utils.voxelgrid import VoxelGrid


def download_checkpoint(path_to_model):
    print('Downloading E2VID checkpoint to {} ...'.format(path_to_model))
    e2vid_model = urllib.request.urlopen('http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar')
    with open(path_to_model, 'w+b') as f:
        f.write(e2vid_model.read())
    print('Done with downloading!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Image reconstruction')
    parser.add_argument('--h5file', help='Path to h5 file containing events for reconstruction.', default='')
    parser.add_argument('-c', '--path_to_model', type=str,
                        help='path to the model weights',
                        default='reconstruction/pretrained/E2VID_lightweight.pth.tar')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--gpu_id',  type=int, default=0)
    parser.add_argument('--freq_hz', '-fhz', type=int, default=0, help='Frequency for saving the reconstructed images from events')
    parser.add_argument('--timestamps_file', '-tsf', help='Path to txt file containing image reconstruction timestamps')
    parser.add_argument('--upsample_rate', '-u', type=int, default=1, help='Multiplies the number of reconstructions, which effectively lowers the time window of events for E2VID. These intermediate reconstructions will not be saved to disk.')
    parser.add_argument('--verbose', '-v',  action='store_true', help='Verbose output')
    parser.add_argument('--index_by_order', '-i',  action='store_true', help='Index reconstrutions with 0,1,2,3...')

    set_inference_options(parser)

    args = parser.parse_args()

    # Data loader
    if not os.path.isfile(args.h5file):
        print('h5 file not provided')
        exit()

    h5_path = Path(args.h5file)
    freq_hz = args.freq_hz
    timestamp_provider = None
    if freq_hz > 0:
        timestamp_provider = TimestampProviderRate(freq_hz)
    else:
        timestamps_file = Path(args.timestamps_file)
        assert timestamps_file.exists()
        timestamp_provider = TimestampProviderFile(timestamps_file)
    data_provider = DataProvider(h5_path, height=args.height, width=args.width, timestamp_provider=timestamp_provider)

    # Load model to device
    if not os.path.isfile(args.path_to_model):
        download_checkpoint(args.path_to_model)
    assert os.path.isfile(args.path_to_model)
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu, args.gpu_id)
    model = model.to(device)
    model.eval()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    else:
        assert os.path.isdir(args.output_folder)

    image_reconstructor = ImageReconstructor(model, args.height, args.width, model.num_bins, args)
    print('== Image reconstruction == ')
    print('Image size: {}x{}'.format(args.height, args.width))
    print('Will write images to: {}'.format(os.path.join(args.output_folder, args.dataset_name)))
    grid = VoxelGrid(model.num_bins, args.width, args.height, upsample_rate=args.upsample_rate)
    pbar = tqdm.tqdm(total=len(data_provider))
    for j, events in enumerate(data_provider):
        if events.events.size > 0:
            sliced_events = grid.event_slicer(events.events, events.t_reconstruction)
            for i in range(len(sliced_events)):
                event_grid, _ = grid.events_to_voxel_grid(sliced_events[i])
                event_tensor = torch.from_numpy(event_grid)
                if i== len(sliced_events) - 1:
                    index = j if args.index_by_order else int(events.t_reconstruction)*1000
                    image_reconstructor.update_reconstruction(event_tensor, index, save=True)
                    pbar.update(1)
                else:
                    image_reconstructor.update_reconstruction(event_tensor)
