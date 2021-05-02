from pathlib import Path

from metavision_core.event_io import EventsIterator

from data.format import Events

def ev_generator(rawfile: Path, delta_t_ms: int=1000) -> Events:
    assert rawfile.exists()
    assert rawfile.suffix == '.raw'

    delta_t_us = delta_t_ms * 1000
    for ev in EventsIterator(str(rawfile), delta_t=delta_t_us):
        events = Events(
                ev['x'].astype('uint16'),
                ev['y'].astype('uint16'),
                ev['p'].astype('uint8'),
                ev['t'].astype('int64'))
        yield events
