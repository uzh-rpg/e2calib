from abc import abstractmethod
import numpy as np

from data.format import Events

class EventAccumulator:
    def __init__(self):
        self.x = list()
        self.y = list()
        self.p = list()
        self.t = list()

    @abstractmethod
    def add_event(self, event):
        pass

    def get_events(self):
        events = Events(
                np.asarray(self.x, dtype='uint16'),
                np.asarray(self.y, dtype='uint16'),
                np.asarray(self.p, dtype='uint8'),
                np.asarray(self.t, dtype='int64'))
        return events

class EventAccumulatorRos(EventAccumulator):
    # overwrite abstract method
    def add_event(self, event):
        self.x.append(event.x)
        self.y.append(event.y)
        self.p.append(int(event.polarity))
        # floor to microseconds.
        self.t.append(event.ts.to_nsec()//1000)

class EventAccumulatorPocolog(EventAccumulator):
    # overwrite abstract method
    def add_event(self, event):
        self.x.append(event.x)
        self.y.append(event.y)
        self.p.append(int(event.polarity))
        self.t.append(event.ts.to_microseconds())
