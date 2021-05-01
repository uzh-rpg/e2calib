from pathlib import Path

from metavision_core.event_io import EventsIterator

def ev_generator(rawfile: Path, delta_t_ms: int=1000):
    assert rawfile.exists()
    assert rawfile.suffix == '.raw'

    delta_t_us = delta_t_ms * 1000
    for ev in EventsIterator(str(rawfile), delta_t=delta_t_us):
        out = {
            'x': ev['x'].astype('uint16'),
            'y': ev['y'].astype('uint16'),
            'p': ev['p'].astype('uint8'),
            't': ev['t'].astype('int64'),
        }
        yield out

if __name__ == '__main__':
    testfile = Path('/home/mathias/Downloads/data.raw')

    for event_slice in ev_generator(testfile):
        print(event_slice['t'])
