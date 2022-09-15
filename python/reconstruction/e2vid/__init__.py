from . import base
from . import model
from . import options
from . import utils
from . import image_reconstructor

import torch
import urllib
import os


def download_checkpoint(path_to_model):
    print('Downloading E2VID checkpoint to {} ...'.format(path_to_model))
    e2vid_model = urllib.request.urlopen('http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar')
    with open(path_to_model, 'w+b') as f:
        f.write(e2vid_model.read())
    print('Done with downloading!')


class E2VID:
    def __init__(self, args):
        os.makedirs(os.path.dirname(args.path_to_model), exist_ok=True)
        if not os.path.isfile(args.path_to_model):
            download_checkpoint(args.path_to_model)
        assert os.path.isfile(args.path_to_model)
        self.model = utils.loading_utils.load_model(args.path_to_model)
        self.device = utils.loading_utils.get_device(args.use_gpu, args.gpu_id)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.image_reconstructor = image_reconstructor.ImageReconstructor(
            self.model, args.height, args.width, self.model.num_bins, args
        )

    def __call__(self, voxel_grid):
        assert len(voxel_grid.shape) == 3
        assert voxel_grid.shape[0] == 5

        event_tensor = torch.from_numpy(voxel_grid)
        image = self.image_reconstructor.update_reconstruction(event_tensor)

        return image

