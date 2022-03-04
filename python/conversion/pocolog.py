from pathlib import Path

# Pocolog bindings
from pocolog_pybind import *

import tqdm
import warnings
import numpy as np

from data.accumulator import EventAccumulatorPocolog

def ev_generator(logpath: Path, delta_t_ms: int=1000, topic: str='/dvs/events'):
    assert logpath.exists()
    assert logpath.suffix == '.log'

    delta_t_ns = delta_t_ms * 10**6

    t_ev_acc_end_ns = None
    ev_acc = EventAccumulatorPocolog()

    multi_file_index = pocolog.MultiFileIndex()
    multi_file_index.create_index([str(logpath)])
    streams = multi_file_index.get_all_streams()
    stream = streams[topic]

    init = False
    last_time = 0
    pbar = tqdm.tqdm(total=stream.get_size())
    for t in range(stream.get_size()):
        value = stream.get_sample(t)
        py_value = value.cast(recursive=True)
        value.destroy()
        # Chech if the events array has at least one event
        if len(py_value['events']) is 0:
            continue

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
                ev_acc = EventAccumulatorPocolog()
                ev_acc.add_event(event)
        pbar.update(1)

