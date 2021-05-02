from pathlib import Path
import weakref

import h5py
import numpy as np

from data.format import Events

class EventAccumulator:
    def __init__(self):
        self.x = list()
        self.y = list()
        self.p = list()
        self.t = list()

    def add_events(self, events: Events):
        self.x.append(events.x)
        self.y.append(events.y)
        self.p.append(events.p)
        self.t.append(events.t)

    def get_t_final(self):
        return self.t[-1][-1]

    def get_events(self) -> Events:
        events = Events(
                np.asarray(self.x),
                np.asarray(self.y),
                np.asarray(self.p),
                np.asarray(self.t))
        return events


class H5GeneratorAbstract:
    def __init__(self, filepath: Path):
        assert filepath.is_file()
        assert filepath.name.endswith('.h5')
        self.h5f = h5py.File(str(filepath), 'r')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class DataProvider(H5GeneratorAbstract):
    def __init__(self, h5file: Path, reconstruction_frequency_hz: int=5):
        super().__init__(h5file)
        assert reconstruction_frequency_hz > 0

        self.delta_t_us = int(1/reconstruction_frequency_hz*10**6)
        self.t_start_us = self.h5f['t'][0]
        self.t_start_current_us = self.t_start_us
        self.t_end_us = self.h5f['t'][-1]

        self.idx_t0 = 0

        self.num_events = self.h5f['t'].size
        self.read_step = 10**6

        self.ev_accumulator = EventAccumulator()
        self.raise_stop = False

    def __next__(self) -> Events:
        while True:
            if self.raise_stop:
                raise StopIteration
            idx_t1 = self.idx_t0 + self.read_step
            if idx_t1 >= self.num_events:
                self.raise_stop = True
                idx_t1 = self.num_events + 1

            # better idea:
            # read only time, when overshoot, generate events dataclass object from accumulator.
            # retrieve the range of interest.
            # reset such that we remove lists except last entry (will be used for next iteration)
            # also need to check everytime, before adding events, if current accumlation has enough events

            read_events = Events(
                    self.h5f['x'][self.idx_t0:idx_t1],
                    self.h5f['y'][self.idx_t0:idx_t1],
                    self.h5f['p'][self.idx_t0:idx_t1],
                    self.h5f['t'][self.idx_t0:idx_t1])
            self.ev_accumulator.add_events(read_events)
            #print(self.ev_accumulator.get_t_final())
            #print(f'num_list = {self.ev_accumulator.get_num_lists()}')

            t_end_current_us = min(self.t_start_current_us + self.delta_t_us, self.t_end_us)
            if self.ev_accumulator.get_t_final() >= t_end_current_us:
                pass

            self.idx_t0 = idx_t1
            return read_events
