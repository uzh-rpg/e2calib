from pathlib import Path

import numpy as np
import tqdm
import rosbag

from data.accumulator import EventAccumulatorRos

def ev_generator(bagpath: Path, delta_t_ms: int=1000, topic: str='/dvs/events'):
    assert bagpath.exists()
    assert bagpath.suffix == '.bag'

    delta_t_ns = delta_t_ms * 10**6

    t_ev_acc_end_ns = None
    ev_acc = EventAccumulatorRos()

    init = False
    last_time = 0
    with rosbag.Bag(str(bagpath), 'r') as bag:
        pbar = tqdm.tqdm(total=bag.get_message_count(topic))
        for topic, msg, ros_time in bag.read_messages(topics=[topic]):
            if not init:
                init = True
                t_start_ns = msg.events[0].ts.to_nsec()
                t_ev_acc_end_ns = t_start_ns + delta_t_ns
            for event in msg.events:
                time = event.ts.to_nsec()
                assert time >= last_time, 'event timestamps must be equal or greater than the previous one'
                last_time = time
                if time < t_ev_acc_end_ns:
                    ev_acc.add_event(event)
                else:
                    events = ev_acc.get_events()
                    yield events
                    t_ev_acc_end_ns = t_ev_acc_end_ns + delta_t_ns
                    ev_acc = EventAccumulatorRos()
                    ev_acc.add_event(event)
            pbar.update(1)
