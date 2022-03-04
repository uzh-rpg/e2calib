import argparse
from pathlib import Path

import conversion.format
import conversion.h5writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert events to h5 format to prepare for calibration.')
    parser.add_argument('input_file', help='Path to file which will be converted to hdf5 format.')
    parser.add_argument('--output_file', '-o', default="", help='Output path for h5 file. Default: Input path but with h5 suffix.')
    parser.add_argument('--topic', '-t', default='/dvs/events', help='Topic name for events if input file is a rosbag(ROS) or pocolog(ROCK).')

    args = parser.parse_args()

    input_file = Path(args.input_file)
    assert input_file.exists()
    if args.output_file:
        output_file = Path(args.output_file)
        assert output_file.suffix == '.h5'
    else:
        output_file = Path(input_file).parent / (input_file.stem + '.h5')
    assert not output_file.exists(), f"{output_file} already exists."

    topic = args.topic

    event_generator = conversion.format.get_generator(input_file, delta_t_ms=1000, topic=topic)
    h5writer = conversion.h5writer.H5Writer(output_file)

    for events in event_generator():
        h5writer.add_data(events)
