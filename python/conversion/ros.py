from pathlib import Path

import numpy as np
import rosbag

from dvs_msgs.msg import EventArray, Event

class EventPackage:
    def __init__(self):
        self.x = list()
        self.y = list()
        self.p = list()
        self.t = list()

    def add_event(self, event: Event):
        self.x.append(event.x)
        self.y.append(event.y)
        self.p.append(int(event.polarity))
        self.t.append(event.ts.to_nsec())

    def get_dict(self):
        out = {
            'x': np.asarray(self.x, dtype='uint16'),
            'y': np.asarray(self.y, dtype='uint16'),
            'p': np.asarray(self.p, dtype='uint8'),
            't': np.asarray(self.t, dtype='int64'),
        }
        return out


def ev_generator(bagpath: Path, delta_t_ms: int=1000, topic: str='/dvs/events'):
    assert bagpath.exists()
    assert bagpath.suffix == '.bag'

    delta_t_us = delta_t_ms * 1000
    delta_t_ns = delta_t_us * 1000

    t_ev_package_end_ns = None
    ev_package = EventPackage()

    init = False
    with rosbag.Bag(str(bagpath), 'r') as bag:
        for topic, msg, ros_time in bag.read_messages(topics=[topic]):
            if not init:
                init = True
                t_start_ns = msg.events[0].ts.to_nsec()
                t_ev_package_end_ns = t_start_ns + delta_t_ns
            for event in msg.events:
                time = event.ts.to_nsec()
                if time < t_ev_package_end_ns:
                    ev_package.add_event(event)
                else:
                    out = ev_package.get_dict()
                    yield out
                    t_ev_package_end_ns = t_ev_package_end_ns + delta_t_ns
                    ev_package = EventPackage()
                    ev_package.add_event(event)

if __name__ == '__main__':
    testfile = Path('/home/mathias/Downloads/indoor_45_16_davis.bag')

    for event_slice in ev_generator(testfile, topic='/dvs/events'):
        print(f"t_s = {event_slice['t'][0]}, t_e = {event_slice['t'][-1]}")
