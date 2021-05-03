# for Python 3.6: pip3 install dataclasses
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Events:
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    t: np.ndarray

    size: int = field(init=False)

    def __post_init__(self):
        assert self.x.dtype == np.uint16
        assert self.y.dtype == np.uint16
        assert self.p.dtype == np.uint8
        assert self.t.dtype == np.int64

        assert self.x.shape == self.y.shape == self.p.shape == self.t.shape
        assert self.x.ndim == 1

        # Without the frozen option, we could just do: self.size = self.x.size
        super().__setattr__('size', self.x.size)

        if self.size > 0:
            assert np.max(self.p) <= 1

@dataclass(frozen=True)
class EventsForReconstruction:
    events: Events

    width: int
    height: int

    t_reconstruction: int

    def __post_init__(self):
        assert self.height > 0
        assert self.width > 0

        if self.events.size > 0:
            assert np.max(self.events.t) <= self.t_reconstruction
            assert np.max(self.events.y) < self.height
            assert np.max(self.events.x) < self.width


if __name__ == '__main__':
    # Example usage.
    events = EventsForReconstruction(
            Events(
                x=np.array([0, 1], dtype='uint16'),
                y=np.array([1, 2], dtype='uint16'),
                p=np.array([0, 0], dtype='uint8'),
                t=np.array([0, 5], dtype='int64')),
            width=2,
            height=3,
            t_reconstruction=6)
    print(events)
