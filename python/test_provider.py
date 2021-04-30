import argparse
from pathlib import Path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_raw', help='Path to prophesee raw file')

    args = parser.parse_args()

    data_raw_path = Path(args.data_raw)
