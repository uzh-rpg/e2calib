import argparse
from pathlib import Path


from data.provider import DataProvider
from data.rectimestamps import TimestampProviderFile, TimestampProviderRate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5file', help='Path to h5 file containing events for reconstruction.')
    parser.add_argument('height', type=int, help='Height of image sensor')
    parser.add_argument('width', type=int, help='Width of image sensor')
    parser.add_argument('--freq_hz', '-fhz', type=int, default=0, help='Frequency for reconstructing frames from events')
    parser.add_argument('--timestamps_file', '-tsf', help='Path to txt file containing image reconstruction timestamps')

    args = parser.parse_args()

    h5_path = Path(args.h5file)
    freq_hz = args.freq_hz
    height = args.height
    width = args.width

    timestamp_provider = None
    if freq_hz > 0:
        timestamp_provider = TimestampProviderRate(freq_hz)
    else:
        timestamps_file = Path(args.timestamps_file)
        assert timestamps_file.exists()
        timestamp_provider = TimestampProviderFile(timestamps_file)

    data_provider = DataProvider(h5_path, height=height, width=width, timestamp_provider=timestamp_provider)

    for events in data_provider:
        if events.events.size > 0:
            print(events.events.t[0])
            print(events.events.t[-1])
            print(events.t_reconstruction)
            print('------------------------')
