#!/usr/bin/env python3
"""Combine RGB and event-reconstructed frames into a single ROS1 bag for Kalibr.

Usage:
    python frames_to_rosbag.py --rgb_dir <dir> --event_dir <dir> --output <bag>

Frame filenames must be nanosecond timestamps (e.g. 1779057492978408620.png).
Topics written:
    /cam0/image_raw  — RGB frames      (bgr8)
    /cam1/image_raw  — Event frames    (mono8)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros1_noetic import (
    builtin_interfaces__msg__Time as Time,
    std_msgs__msg__Header as Header,
    sensor_msgs__msg__Image as Image,
)


def make_image_msg(img_bgr: np.ndarray, stamp_ns: int, frame_id: str = '') -> Image:
    sec  = int(stamp_ns // 1_000_000_000)
    nsec = int(stamp_ns  % 1_000_000_000)
    header = Header(seq=0, stamp=Time(sec=sec, nanosec=nsec), frame_id=frame_id)

    if img_bgr.ndim == 2:
        encoding = 'mono8'
        data = img_bgr.tobytes()
        step = img_bgr.shape[1]
    else:
        encoding = 'bgr8'
        data = img_bgr.tobytes()
        step = img_bgr.shape[1] * 3

    return Image(
        header=header,
        height=img_bgr.shape[0],
        width=img_bgr.shape[1],
        encoding=encoding,
        is_bigendian=0,
        step=step,
        data=np.frombuffer(data, dtype=np.uint8),
    )


def load_frames(folder: Path):
    """Return sorted list of (timestamp_ns, path) from a folder of PNG files."""
    frames = []
    for p in sorted(folder.glob('*.png')):
        try:
            ts = int(p.stem)
            frames.append((ts, p))
        except ValueError:
            print(f"  Skipping {p.name} — filename is not a timestamp")
    return frames


def write_bag(rgb_dir: Path, event_dir: Path, output_bag: Path,
              rgb_topic: str, event_topic: str):
    typestore = get_typestore(Stores.ROS1_NOETIC)

    rgb_frames   = load_frames(rgb_dir)
    event_frames = load_frames(event_dir)
    print(f"RGB frames   : {len(rgb_frames)}")
    print(f"Event frames : {len(event_frames)}")

    if not rgb_frames and not event_frames:
        print("ERROR: no frames found.")
        sys.exit(1)

    # Merge and sort all messages by timestamp
    all_msgs = []
    for ts, path in rgb_frames:
        all_msgs.append((ts, path, rgb_topic, False))
    for ts, path in event_frames:
        all_msgs.append((ts, path, event_topic, True))
    all_msgs.sort(key=lambda x: x[0])

    rgb_conn_id   = None
    event_conn_id = None

    with Writer(str(output_bag)) as writer:
        if rgb_frames:
            rgb_conn_id = writer.add_connection(rgb_topic, Image.__msgtype__,
                                                typestore=typestore)
        if event_frames:
            event_conn_id = writer.add_connection(event_topic, Image.__msgtype__,
                                                  typestore=typestore)

        for i, (ts_ns, path, topic, is_event) in enumerate(all_msgs):
            if is_event:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)

            if img is None:
                print(f"  WARNING: could not read {path}, skipping")
                continue

            msg  = make_image_msg(img, ts_ns)
            conn = event_conn_id if is_event else rgb_conn_id
            rawdata = typestore.serialize_ros1(msg, Image.__msgtype__)
            writer.write(conn, ts_ns, rawdata)

            if i % 200 == 0:
                print(f"  {i+1}/{len(all_msgs)} written...", end='\r')

    print(f"\nBag saved → {output_bag}")
    print(f"Topics: {rgb_topic}  |  {event_topic}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rgb_dir',   required=True,  help='Folder of RGB PNG frames')
    parser.add_argument('--event_dir', required=True,  help='Folder of event PNG frames')
    parser.add_argument('--output',    required=True,  help='Output .bag file path')
    parser.add_argument('--rgb_topic',   default='/cam0/image_raw')
    parser.add_argument('--event_topic', default='/cam1/image_raw')
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        output.unlink()

    write_bag(
        rgb_dir   = Path(args.rgb_dir),
        event_dir = Path(args.event_dir),
        output_bag = output,
        rgb_topic  = args.rgb_topic,
        event_topic = args.event_topic,
    )
