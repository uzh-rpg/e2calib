from pathlib import Path
import warnings

import numpy as np

from conversion.prophesee_utils import load_td_data

from data.format import Events

def ev_generator(rawfile: Path, delta_t_ms: int=1000) -> Events:
    assert rawfile.exists()
    assert rawfile.suffix == '.dat'

    delta_t_us = delta_t_ms * 1000
    cd_data = load_td_data(rawfile)
    for t in range(np.min(cd_data['t']), np.max(cd_data['t']), delta_t_us):
        ev = cd_data[cd_data['t']>=t]
        ev = ev[ev['t']<t+delta_t_us]
        is_sorted = np.all(ev['t'][:-1] <= ev['t'][1:])
        if not is_sorted:
            warnings.warn('Event timestamps are not sorted.', stacklevel=2)
        events = Events(
                ev['x'].astype('uint16'),
                ev['y'].astype('uint16'),
                ev['p'].astype('uint8'),
                ev['t'].astype('int64'))
        yield events
