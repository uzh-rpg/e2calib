from pathlib import Path

# Pocolog bindings
from pocolog_pybind import *

import numpy as np

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
        self.t.append(event.ts.to_microseconds())

    def get_events(self):
        events = Events(
                np.asarray(self.x, dtype='uint16'),
                np.asarray(self.y, dtype='uint16'),
                np.asarray(self.p, dtype='uint8'),
                np.asarray(self.t, dtype='int64'))
        return events


def ev_generator(logpath: Path, delta_t_ms: int=1000, topic: str='/dvs/events'):
    assert logpath.exists()
    assert logpath.suffix == '.log'

    delta_t_ns = delta_t_ms * 10**6

    t_ev_acc_end_ns = None
    ev_acc = EventAccumulator()

    multi_file_index = pocolog.MultiFileIndex()
    multi_file_index.create_index([str(logpath)])
    streams = multi_file_index.get_all_streams()
    stream = streams[topic]

    init = False
    last_time = 0
    for t in range(stream.get_size()):
        value = stream.get_sample(t)
        py_value = value.cast(recursive=True)
        value.destroy()
        if not init:
            init = True
            t_start_ns = py_value['events'][0].ts.to_microseconds()*1e03
            t_ev_acc_end_ns = t_start_ns + delta_t_ns
        for event in py_value['events']:
            time = event.ts.to_microseconds()*1e03
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

