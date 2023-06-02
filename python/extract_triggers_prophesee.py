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


def get_reconstruction_timestamps(time: np.ndarray, pol: np.ndarray, time_offset_us: int=0):
    assert 0 <= pol.max() <= 1
    assert np.all(np.abs(np.diff(pol)) == 1), 'polarity must alternate from trigger to trigger'

    assert pol[0] == 1, 'first ext trigger polarity must be positive'
    assert pol[-1] == 0, 'last ext trigger polarity must be negative'

    rising_ts = time[pol==1]
    falling_ts = time[pol==0]

    return {
        'middle': time_offset_us + ((rising_ts + falling_ts) / 2).astype('int64'),
        'rising': time_offset_us + rising_ts.astype('int64'),
        'falling': time_offset_us + falling_ts.astype('int64'),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read trigger data from prophesee raw file')
    parser.add_argument('rawfile')
    parser.add_argument('output_file', help='Path to output text file with timestamps for reconstruction.')
    parser.add_argument('--use_timestamp', choices=["rising", "middle", "falling", "all"], default="rising",
            help='Whether to take the timestamp of the rising, falling, or the middle timestamp.')
    parser.add_argument('--time_offset_us', '-offset', type=int, default=0,
            help='Add a constant time offset to the timestamp for reconstruction. '
            'Can be useful to compensate for a constant exposure time of global shutter cameras.')

    args = parser.parse_args()

    rawfile = Path(args.rawfile)
    outfile_base = Path(args.output_file)
    assert not outfile_base.exists()
    assert outfile_base.suffix == '.txt'

    triggers = get_ext_trigger_timestamps(rawfile)
    timestamps = get_reconstruction_timestamps(triggers['t'], triggers['p'], args.time_offset_us)

    timestamps_keys = [args.use_timestamp] if args.use_timestamp != "all" else ["rising", "falling", "middle"]
    for k in timestamps_keys:
        outfile = outfile_base
        if args.use_timestamp == "all" and k != "middle":
            # If "all" are asked, save "middle" with the provided filename, and
            # <filename>_rising.txt or <filename>_falling.txt for the other two
            outfile = outfile_base.with_name(outfile_base.stem + f"_{k}" + outfile_base.suffix)
        print(f"Writing {k} timestamps to {outfile}")
        np.savetxt(str(outfile), timestamps[k], '%i')
