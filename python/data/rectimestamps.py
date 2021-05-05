from pathlib import Path

import numpy as np


class TimestampProviderBase:
    def __init__(self):
        self.initialized = False

    def initialize(self, t_start_us: int, t_end_us: int):
        raise NotImplementedError

    def __iter__(self):
        assert self.initialized
        return self

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class TimestampProviderRate(TimestampProviderBase):
    def __init__(self, reconstruction_frequency_hz: int=5):
        super().__init__()
        assert reconstruction_frequency_hz > 0

        self.delta_t_us = int(1/reconstruction_frequency_hz*10**6)

        # We still need to initialize the following values
        self.t_end_us = None
        self._length = None
        self.t_rec_us = None

    def initialize(self, t_start_us: int, t_end_us: int):
        assert not self.initialized
        assert t_end_us > t_start_us
        self.t_end_us = t_end_us
        self._length = (self.t_end_us - t_start_us)//self.delta_t_us + 1

        # We add delta_t_us before we return it.
        self.t_rec_us = t_start_us - self.delta_t_us

        self.initialized = True

    def __len__(self):
        assert self.initialized
        return self._length

    def __next__(self):
        assert self.initialized
        self.t_rec_us += self.delta_t_us

        if self.t_rec_us > self.t_end_us:
            raise StopIteration

        return self.t_rec_us


class TimestampProviderFile(TimestampProviderBase):
    def __init__(self, timestamp_file: Path):
        super().__init__()
        self.initialized = True

        assert timestamp_file.exists()
        assert timestamp_file.suffix == '.txt'
        self.timestamps = np.loadtxt(str(timestamp_file), dtype=np.int64)

        self.idx = -1

    def initialize(self, t_start_us: int, t_end_us: int):
        pass

    def __len__(self):
        return len(self.timestamps)

    def __next__(self):
        self.idx += 1
        if self.idx == self.__len__():
            raise StopIteration
        return self.timestamps[self.idx]
