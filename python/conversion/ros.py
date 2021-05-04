from pathlib import Path

import numpy as np
import rosbag

from data.format import Events

class EventAccumulator:
    def __init__(self):
        self.x = list()
        self.y = list()
        self.p = list()
        self.t = list()

    def add_event(self, event):
        self.x.append(event.x)
        self.y.append(event.y)
        self.p.append(int(event.polarity))
        # floor to microseconds.
        self.t.append(event.ts.to_nsec()//1000)

    def get_events(self):
        events = Events(
                np.asarray(self.x, dtype='uint16'),
                np.asarray(self.y, dtype='uint16'),
                np.asarray(self.p, dtype='uint8'),
                np.asarray(self.t, dtype='int64'))
        return events


def ev_generator(bagpath: Path, delta_t_ms: int=1000, topic: str='/dvs/events'):
    assert bagpath.exists()
    assert bagpath.suffix == '.bag'

    delta_t_ns = delta_t_ms * 10**6

    t_ev_acc_end_ns = None
    ev_acc = EventAccumulator()

    init = False
    last_time = 0
    with rosbag.Bag(str(bagpath), 'r') as bag:
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
                    ev_acc = EventAccumulator()
                    ev_acc.add_event(event)
