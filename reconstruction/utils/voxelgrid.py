import sys
import numpy as np
from utils.data import Events

class VoxelGrid:
    def __init__(self, num_bins: int=5, width: int=640, height: int=480, upsample_rate: int=1):
        assert num_bins > 1
        assert height > 0
        assert width > 0
        self.num_bins = num_bins
        self.width = width
        self.height = height
        self.upsample_rate = upsample_rate

    def event_slicer(self, events: Events):
        sliced_events = []
        t_start = events.t[0]
        t_end = events.t[-1]
        window_time = (t_end - t_start + 1)/self.upsample_rate
        indices = [0]
        for i in range(1, self.upsample_rate):
            indices.append( np.where(events.t > i*window_time+t_start)[0][0])
        indices.append(len(events.t)) # Add the last time timestamp
        
        # The format here is used in e2vid voxel grid
        for i in range(0, self.upsample_rate):
            sliced_events.append(np.stack((
                np.asarray(events.t[indices[i]:indices[i+1]], dtype="float32"),
                np.asarray(events.x[indices[i]:indices[i+1]], dtype="float32"),
                np.asarray(events.y[indices[i]:indices[i+1]], dtype="float32"),
                np.asarray(events.p[indices[i]:indices[i+1]], dtype="float32"))).T)
        return sliced_events

    def events_to_voxel_grid(self, events):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        """

        assert(events.shape[1] == 4)

        voxel_grid = np.zeros((self.num_bins, self.height, self.width), np.float32).ravel()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events[-1, 0]
        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events[:, 0] = (self.num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
        ts = events[:, 0]
        xs = events[:, 1].astype(np.int)
        ys = events[:, 2].astype(np.int)
        pols = events[:, 3]
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = ts.astype(np.int)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        valid_indices = tis < self.num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * self.width +
                tis[valid_indices] * self.width * self.height, vals_left[valid_indices])

        valid_indices = (tis + 1) < self.num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * self.width +
                (tis[valid_indices] + 1) * self.width * self.height, vals_right[valid_indices])

        voxel_grid = np.reshape(voxel_grid, (self.num_bins, self.height, self.width))
        return voxel_grid, last_stamp


if __name__ == '__main__':
    events = Events(
            x=np.array([0, 1, 5, 3, 4 ,7], dtype='uint16'),
            y=np.array([1, 2, 4, 3, 4, 1], dtype='uint16'),
            p=np.array([0, 0, 1, 1, 0, 1], dtype='uint8'),
            t=np.array([5, 9, 11, 17, 27, 30], dtype='int64'),
            width=8,
            height=5,
            t_reconstruction=31)
    grid_repr = VoxelGrid(5, events.width, events.height, upsample_rate=2)
    sliced_events = grid_repr.event_slicer(events)
    voxel_grid = []
    for i in range(len(sliced_events)):
        voxel_grid.append(grid_repr.events_to_voxel_grid(sliced_events[i]))
