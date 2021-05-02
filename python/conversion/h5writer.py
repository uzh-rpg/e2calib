from pathlib import Path
import weakref

import h5py
import numpy as np


class H5Writer:
    def __init__(self, outfile: Path):
        assert not outfile.exists(), str(outfile)
        self.h5f = h5py.File(str(outfile), 'w')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

        # create hdf5 datasets
        shape = (2**16,)
        maxshape = (None,)
        compression = 'lzf'
        self.h5f.create_dataset('x', shape=shape, dtype='u2', chunks=shape, maxshape=maxshape, compression=compression)
        self.h5f.create_dataset('y', shape=shape, dtype='u2', chunks=shape, maxshape=maxshape, compression=compression)
        self.h5f.create_dataset('p', shape=shape, dtype='u1', chunks=shape, maxshape=maxshape, compression=compression)
        self.h5f.create_dataset('t', shape=shape, dtype='i8', chunks=shape, maxshape=maxshape, compression=compression)
        self.row_idx = 0

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def add_data(self, x: np.ndarray, y: np.ndarray, pol: np.ndarray, t: np.ndarray):
        assert x.ndim == y.ndim == pol.ndim == t.ndim == 1
        assert x.shape == y.shape == pol.shape == t.shape
        assert x.size > 0

        x = x.astype('uint16')
        y = y.astype('uint16')
        pol = pol.astype('uint8')
        t = t.astype('int64')

        current_size = x.size
        new_size = self.row_idx + current_size
        self.h5f['x'].resize(new_size, axis=0)
        self.h5f['y'].resize(new_size, axis=0)
        self.h5f['p'].resize(new_size, axis=0)
        self.h5f['t'].resize(new_size, axis=0)

        self.h5f['x'][self.row_idx:new_size] = x
        self.h5f['y'][self.row_idx:new_size] = y
        self.h5f['p'][self.row_idx:new_size] = pol
        self.h5f['t'][self.row_idx:new_size] = t

        self.row_idx = new_size
