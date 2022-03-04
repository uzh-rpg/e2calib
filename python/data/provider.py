import gc
from pathlib import Path
from typing import Optional
import weakref

import h5py
import numpy as np

from data.format import Events, EventsForReconstruction
from data.rectimestamps import TimestampProviderBase


class SharedEventBuffer:
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
        if len(self.t) == 0:
            return None
        return self.t[-1][-1]

    def get_events(self) -> Events:
        events = Events(
                np.concatenate(self.x),
                np.concatenate(self.y),
                np.concatenate(self.p),
                np.concatenate(self.t))
        return events

    def clean(self, t_cutoff: int):
        # Remove event arrays if they only contain timestamps lower than t_cutoff.
        del_idx = 0
        for idx in range(0, len(self.t)):
            if self.t[idx][-1] < t_cutoff:
                del_idx = idx + 1
        if del_idx > 0:
            del self.x[:del_idx]
            del self.y[:del_idx]
            del self.p[:del_idx]
            del self.t[:del_idx]
            gc.collect()


class SharedBufferProducer:
    def __init__(self, h5file: Path, shared_ev_buffer: SharedEventBuffer):
        assert h5file.is_file()
        assert h5file.name.endswith('.h5')
        self.h5f = h5py.File(str(h5file), 'r')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        # Shared buffer (akin to producer-consumer pattern)
        self.shared_ev_buffer = shared_ev_buffer

        # Number of events to be read from h5 per iteration.
        self.read_step = 10**7

        self.idx_0 = 0
        self.num_events = self.h5f['t'].size

        self._done = False

        self.t_start_us = self.h5f['t'][0]
        self.t_end_us = self.h5f['t'][-1]

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def write_to_shared_buffer(self):
        if self._done:
            print('No more events to be read')
            return
        idx_1 = self.idx_0 + self.read_step
        self._done = idx_1 >= self.num_events
        if self._done:
            idx_1 = self.num_events
        if idx_1 > self.idx_0:
            read_events = Events(
                    self.h5f['x'][self.idx_0:idx_1],
                    self.h5f['y'][self.idx_0:idx_1],
                    self.h5f['p'][self.idx_0:idx_1],
                    self.h5f['t'][self.idx_0:idx_1])
            self.shared_ev_buffer.add_events(read_events)
        self.idx_0 = idx_1

    def done(self):
        return self._done

    def get_t_start_us(self):
        return self.t_start_us

    def get_t_end_us(self):
        return self.t_end_us


class SharedBufferConsumer:
    def __init__(self, h5file: Path):
        # Shared buffer (akin to producer-consumer pattern)
        self.shared_buffer = SharedEventBuffer()
        # We create a producer to fill up the shared buffer, if necessary.
        self.shared_buffer_producer = SharedBufferProducer(h5file, self.shared_buffer)

        # Local buffer stores events directly in  numpy arrays,
        # which is more efficient than list of numpy arrays (shared buffer).
        self.local_buffer = None

        self.last_time = -1

    def get_t_start_us(self):
        return self.shared_buffer_producer.get_t_start_us()

    def get_t_end_us(self):
        return self.shared_buffer_producer.get_t_end_us()

    def get_events_until(self, time: int) -> Optional[Events]:
        # Returns events if successful, None otherwise.

        assert time >= self.last_time, f'time = {time}, last_time={self.last_time}'
        # First ensure that there are enough events in the shared buffer.
        if self.update_shared_buffer(time):
            # Successfull update: Ensure that there are enough events in local buffer.
            self.update_local_buffer(time)
            # Retrieve events.
            self.local_buffer: Events
            indices = np.asarray(np.logical_and(self.local_buffer.t >= self.last_time, self.local_buffer.t < time)).nonzero()
            if indices[0].size == 0:
                retrieved_events = Events(
                        np.array([], dtype=np.uint16),
                        np.array([], dtype=np.uint16),
                        np.array([], dtype=np.uint8),
                        np.array([], dtype=np.int64))
            else:
                retrieved_events = Events(
                        self.local_buffer.x[indices],
                        self.local_buffer.y[indices],
                        self.local_buffer.p[indices],
                        self.local_buffer.t[indices])
            # Update last time for next call.
            self.last_time = time
            return retrieved_events
        # Shared buffer cannot provide enough events anymore.
        self.last_time = time
        return None

    def update_local_buffer(self, time: int):
        # Assumes that the shared buffer is up-to-date.
        assert time <= self.shared_buffer.get_t_final()
        if self.local_buffer is None:
            # Initialize local buffer for the first time.
            self.local_buffer = self.shared_buffer.get_events()
        if time > self.local_buffer.t[-1]:
            # We have to update the local buffer based on the shared buffer.
            self.local_buffer = self.shared_buffer.get_events()
            assert time <= self.local_buffer.t[-1]

    def update_shared_buffer(self, time: int):
        # Return True if shared buffer is up-to-date.

        # Clean shared buffer based on last_time first
        assert time >= self.last_time
        self.shared_buffer.clean(self.last_time)

        # Add more events if time is larger than event timestamps in the shared buffer.
        while self.shared_buffer.get_t_final() is None or time > self.shared_buffer.get_t_final():
            if self.shared_buffer_producer.done():
                return False
            self.shared_buffer_producer.write_to_shared_buffer()
        return True


class DataProvider:
    def __init__(self, h5file: Path, height: int, width: int, timestamp_provider: TimestampProviderBase):
        assert height > 0
        assert width > 0

        self.shared_buffer_consumer = SharedBufferConsumer(h5file)

        self.height = height
        self.width = width

        self.timestamp_provider = timestamp_provider
        self.timestamp_provider.initialize(
                self.shared_buffer_consumer.get_t_start_us(),
                self.shared_buffer_consumer.get_t_end_us())

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.timestamp_provider)

    def __next__(self) -> EventsForReconstruction:
        t_reconstruction_us = next(self.timestamp_provider)

        events = self.shared_buffer_consumer.get_events_until(t_reconstruction_us)
        if events is None:
            raise StopIteration
        return EventsForReconstruction(events, self.width, self.height, t_reconstruction_us)
