import argparse
from pathlib import Path

import numpy as np

from metavision_core.event_io import RawReader


def get_ext_trigger_timestamps(rawfile: Path):
    assert rawfile.exists()
    assert rawfile.suffix == '.raw'

    rawreader = RawReader(str(rawfile))

    while not rawreader.is_done():
        rawreader.load_delta_t(10**5)
    ext_trigger_list = rawreader.get_ext_trigger_events()
    time = ext_trigger_list['t']
    pol = ext_trigger_list['p']

    return {'t': time, 'p': pol}


def get_reconstruction_timestamps(time: np.ndarray, pol: np.ndarray, use_avg_ts: bool=False, time_offset_us: int=0):
    assert 0 <= pol.max() <= 1
    assert np.all(np.abs(np.diff(pol)) == 1), 'polarity must alternate from trigger to trigger'

    timestamps = None
    if use_avg_ts:
        assert pol[0] == 1, 'first ext trigger polarity must be positive'
        assert pol[-1] == 0, 'last ext trigger polarity must be negative'
        rising_ts = time[pol==1]
        falling_ts = time[pol==0]
        timestamps = ((rising_ts + falling_ts)/2).astype('int64')
    else:
        timestamps = time[pol==1]

    timestamps = timestamps + time_offset_us
    return timestamps


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read trigger data from prophesee raw file')
    parser.add_argument('rawfile')
    parser.add_argument('output_file', help='Path to output text file with timestamps for reconstruction.')
    parser.add_argument('--use_average_timestamp', '-avgt', action='store_true',
            help='Take the average timestamp of rising and falling edge for the timestamp. '
            'Otherwise, use the rising edge as timestamp.')
    parser.add_argument('--time_offset_us', '-offset', type=int, default=0,
            help='Add a constant time offset to the timestamp for reconstruction. '
            'Can be useful to compensate for a constant exposure time of global shutter cameras.')

    args = parser.parse_args()

    rawfile = Path(args.rawfile)
    outfile = Path(args.output_file)
    assert not outfile.exists()
    assert outfile.suffix == '.txt'

    triggers = get_ext_trigger_timestamps(rawfile)
    reconstruction_ts = get_reconstruction_timestamps(triggers['t'], triggers['p'], args.use_average_timestamp, args.time_offset_us)
    np.savetxt(str(outfile), reconstruction_ts, '%i')
