import argparse
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
from pathlib import Path
import warnings

import tqdm

from data.provider import DataProvider
from data.rectimestamps import TimestampProviderFile, TimestampProviderRate

import e2vid
from e2vid.options.inference_options import set_inference_options
from e2vid.utils.voxelgrid import VoxelGrid



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Image reconstruction')
    parser.add_argument('--h5file', help='Path to h5 file containing events for reconstruction.', default='')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--freq_hz', '-fhz', type=int, default=0, help='Frequency for saving the reconstructed images from events')
    parser.add_argument('--timestamps_file', '-tsf', help='Path to txt file containing image reconstruction timestamps')
    parser.add_argument('--upsample_rate', '-u', type=int, default=1, help='Multiplies the number of reconstructions, which effectively lowers the time window of events for E2VID. These intermediate reconstructions will not be saved to disk.')
    
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
    reconstructor = e2vid.E2VID(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    else:
        assert os.path.isdir(args.output_folder)

    print('== Image reconstruction == ')
    print('Image size: {}x{}'.format(args.height, args.width))
    print('Will write images to: {}'.format(os.path.join(args.output_folder, args.dataset_name)))
    grid = VoxelGrid(reconstructor.model.num_bins, args.width, args.height, upsample_rate=args.upsample_rate)
    pbar = tqdm.tqdm(total=len(data_provider))
    for events in data_provider:
        if events.events.size > 0:
            sliced_events = grid.event_slicer(events.events, events.t_reconstruction)
            for i in range(len(sliced_events)):
                event_slice = sliced_events[i]
                if event_slice is None:
                    warnings.warn('No events returned. We do not reconstruct for this timestamp')
                else:
                    event_grid, _ = grid.events_to_voxel_grid(sliced_events[i])
                    event_grid = grid.normalize_voxel(event_grid)
                    reconstructor(event_grid)
                if i== len(sliced_events) - 1:
                    rec_ts_nanoseconds = int(events.t_reconstruction)*1000
                    reconstructor.image_reconstructor.save_reconstruction(rec_ts_nanoseconds)
                    pbar.update(1)
