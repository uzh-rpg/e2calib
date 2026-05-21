#!/usr/bin/env python3
"""Convert ROS2 bag to e2calib H5 (events), RGB PNG frames, and RGB timestamps file.

Usage:
    python convert_ros2_bag.py <bag_dir> [output.h5] [--rgb_dir <dir>] [--no_events] [--no_rgb] [--overwrite]

The bag must contain:
  /events  sensor_msgs/msg/PointCloud2  (fields: x, y, t, polarity — all float32)
  /rgb     sensor_msgs/msg/Image

After running this script, reconstruct event frames synchronized to RGB timestamps:
    conda activate e2calib
    python offline_reconstruction.py \\
        --h5file <output.h5> \\
        --timestamps_file <bag_dir>/../rgb_timestamps.txt \\
        --upsample_rate 2 --height 480 --width 640 \\
        --output_folder <bag_dir>/../frames --use_gpu
"""

import sys
import argparse
import numpy as np
import h5py
import cv2
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# PointCloud2 field layout: x f4, y f4, t f4 (seconds, relative), polarity f4
PC2_DTYPE = np.dtype([('x', 'f4'), ('y', 'f4'), ('t', 'f4'), ('polarity', 'f4')])


def save_events(reader, typestore, output_h5: Path, topic: str = '/events', overwrite: bool = False):
    if output_h5.exists():
        if overwrite:
            output_h5.unlink()
        else:
            raise FileExistsError(f"{output_h5} already exists. Use --overwrite to replace it.")

    connections = [c for c in reader.connections if c.topic == topic]
    if not connections:
        available = [c.topic for c in reader.connections]
        raise RuntimeError(f"Topic {topic!r} not found. Available: {available}")

    x_all, y_all, t_all, p_all = [], [], [], []

    print(f"Reading events from {topic} ...")
    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        if msg.width == 0:
            continue

        arr = np.frombuffer(bytes(msg.data), dtype=PC2_DTYPE)
        header_us = int(msg.header.stamp.sec) * 1_000_000 + msg.header.stamp.nanosec // 1000
        t_us = header_us + np.round(arr['t'] * 1_000_000).astype('int64')

        x_all.append(arr['x'].astype('uint16'))
        y_all.append(arr['y'].astype('uint16'))
        p_all.append(arr['polarity'].astype('uint8'))
        t_all.append(t_us)

    if not x_all:
        print("WARNING: no events found on topic, skipping H5 output.")
        return

    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    t = np.concatenate(t_all)
    p = np.concatenate(p_all)

    print(f"Total events : {len(x):,}")
    print(f"Duration     : {(t[-1] - t[0]) / 1e6:.2f} s")

    with h5py.File(str(output_h5), 'w') as f:
        f.create_dataset('x', data=x, dtype='u2', compression='lzf')
        f.create_dataset('y', data=y, dtype='u2', compression='lzf')
        f.create_dataset('p', data=p, dtype='u1', compression='lzf')
        f.create_dataset('t', data=t, dtype='i8', compression='lzf')

    print(f"Events saved → {output_h5}")


def save_rgb_frames(reader, typestore, rgb_dir: Path, timestamps_file: Path, topic: str = '/rgb'):
    connections = [c for c in reader.connections if c.topic == topic]
    if not connections:
        available = [c.topic for c in reader.connections]
        raise RuntimeError(f"Topic {topic!r} not found. Available: {available}")

    rgb_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading RGB frames from {topic} ...")
    timestamps_us = []
    count = 0

    for connection, timestamp, rawdata in reader.messages(connections=connections):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

        ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + msg.header.stamp.nanosec
        ts_us = ts_ns // 1000

        img_data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        encoding = msg.encoding.lower()

        if encoding in ('rgb8',):
            img = img_data.reshape((msg.height, msg.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif encoding in ('bgr8',):
            img = img_data.reshape((msg.height, msg.width, 3))
        elif encoding in ('mono8', '8uc1'):
            img = img_data.reshape((msg.height, msg.width))
        elif encoding in ('bgra8', 'rgba8'):
            img = img_data.reshape((msg.height, msg.width, 4))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR if 'bgr' in encoding else cv2.COLOR_RGBA2BGR)
        else:
            print(f"  Unknown encoding {encoding!r}, attempting raw reshape")
            img = img_data.reshape((msg.height, msg.width, -1))

        fname = rgb_dir / f"{ts_ns:019d}.png"
        cv2.imwrite(str(fname), img)
        timestamps_us.append(ts_us)
        count += 1

    # Save timestamps in microseconds for offline_reconstruction.py --timestamps_file
    np.savetxt(str(timestamps_file), np.array(timestamps_us, dtype=np.int64), fmt='%d')

    print(f"RGB frames saved     : {count} frames → {rgb_dir}")
    print(f"RGB timestamps saved : {count} entries → {timestamps_file}")
    print(f"  (use --timestamps_file {timestamps_file} for synchronized event reconstruction)")


def convert(bag_dir: str, output_h5: str, rgb_dir: str, timestamps_file: str,
            do_events: bool = True, do_rgb: bool = True, overwrite: bool = False):
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with Reader(str(bag_dir)) as reader:
        if do_events:
            save_events(reader, typestore, Path(output_h5), overwrite=overwrite)
        if do_rgb:
            save_rgb_frames(reader, typestore, Path(rgb_dir), Path(timestamps_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag_dir', help='Path to ROS2 bag directory')
    parser.add_argument('output_h5', nargs='?', default='',
                        help='Output H5 file path (default: <bag_dir>/../events.h5)')
    parser.add_argument('--rgb_dir', default='',
                        help='Output folder for RGB PNGs (default: <bag_dir>/../rgb_frames)')
    parser.add_argument('--timestamps_file', default='',
                        help='Output path for RGB timestamps txt (default: <bag_dir>/../rgb_timestamps.txt)')
    parser.add_argument('--no_events', action='store_true', help='Skip event conversion')
    parser.add_argument('--no_rgb', action='store_true', help='Skip RGB extraction')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files')
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    output_h5       = args.output_h5       or str(bag_dir.parent / 'events.h5')
    rgb_dir         = args.rgb_dir         or str(bag_dir.parent / 'rgb_frames')
    timestamps_file = args.timestamps_file or str(bag_dir.parent / 'rgb_timestamps.txt')

    convert(bag_dir, output_h5, rgb_dir, timestamps_file,
            do_events=not args.no_events,
            do_rgb=not args.no_rgb,
            overwrite=args.overwrite)
