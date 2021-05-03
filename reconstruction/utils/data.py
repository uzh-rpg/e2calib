import sys
version = sys.version_info
# assert version[0] >= 3
# assert version[1] >= 7, 'required for dataclasses'
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Events:
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    t: np.ndarray

    width: int
    height: int

    t_reconstruction: int

    def __post_init__(self):
        assert self.x.dtype == np.uint16
        assert self.y.dtype == np.uint16
        assert self.p.dtype == np.uint8
        assert self.t.dtype == np.int64

        assert self.x.shape == self.y.shape == self.p.shape == self.t.shape
        assert self.x.ndim == 1

        assert np.max(self.p) <= 1
        assert np.max(self.t) <= self.t_reconstruction

        assert self.height > 0
        assert np.max(self.y) < self.height
        assert self.width > 0
        assert np.max(self.x) < self.width


if __name__ == '__main__':
    events = Events(
            x=np.array([0, 1], dtype='uint16'),
            y=np.array([1, 2], dtype='uint16'),
            p=np.array([0, 0], dtype='uint8'),
            t=np.array([0, 5], dtype='int64'),
            width=2,
            height=3,
            t_reconstruction=6)
    print(events)
