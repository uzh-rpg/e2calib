import argparse
from pathlib import Path


from data.provider import DataProvider

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('h5file', help='Path to h5 file containing events for reconstruction.')
    parser.add_argument('--freq_hz', '-fhz', type=int, default=5, help='Frequency for reconstructing frames from events')

    args = parser.parse_args()

    h5_path = Path(args.h5file)
    freq_hz = args.freq_hz

    data_provider = DataProvider(h5_path, freq_hz)

    for events in data_provider:
        pass
        #print(events.t[0])
