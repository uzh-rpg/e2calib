from .util import robust_min, robust_max
from .path_utils import ensure_dir
from .loading_utils import get_device
from os.path import join
from math import ceil, floor
from torch.nn import ReflectionPad2d
import numpy as np
import torch
import cv2
from collections import deque
import atexit
import scipy.stats as st
import torch.nn.functional as F


def make_event_preview(events, mode='red-blue', num_bins_to_show=-1):
    # events: [1 x C x H x W] event tensor
    # mode: 'red-blue' or 'grayscale'
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    assert(mode in ['red-blue', 'grayscale'])
    if num_bins_to_show < 0:
        sum_events = torch.sum(events[0, :, :, :], dim=0).detach().cpu().numpy()
    else:
        sum_events = torch.sum(events[0, -num_bins_to_show:, :, :], dim=0).detach().cpu().numpy()

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        b[sum_events > 0] = 255
        r[sum_events < 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = -10.0, 10.0
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)).astype(np.uint8), 0, 255)

    return event_preview


def gkern(kernlen=5, nsig=1.0):
    """Returns a 2D Gaussian kernel array."""
    """https://stackoverflow.com/a/29731818"""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return torch.from_numpy(kernel).float()


class IntensityRescaler:
    """
    Utility class to rescale image intensities to the range [0, 1],
    using (robust) min/max normalization.
    Optionally, the min/max bounds can be smoothed over a sliding window to avoid jitter.
    """

    def __init__(self, options):
        self.auto_hdr = options.auto_hdr
        self.intensity_bounds = deque()
        self.auto_hdr_median_filter_size = options.auto_hdr_median_filter_size
        self.Imin = options.Imin
        self.Imax = options.Imax

    def __call__(self, img):
        """
        param img: [1 x 1 x H x W] Tensor taking values in [0, 1]
        """
        if self.auto_hdr:
            Imin = torch.min(img).item()
            Imax = torch.max(img).item()

            # ensure that the range is at least 0.1
            Imin = np.clip(Imin, 0.0, 0.45)
            Imax = np.clip(Imax, 0.55, 1.0)

            # adjust image dynamic range (i.e. its contrast)
            if len(self.intensity_bounds) > self.auto_hdr_median_filter_size:
                self.intensity_bounds.popleft()

            self.intensity_bounds.append((Imin, Imax))
            self.Imin = np.median([rmin for rmin, rmax in self.intensity_bounds])
            self.Imax = np.median([rmax for rmin, rmax in self.intensity_bounds])

        img = 255.0 * (img - self.Imin) / (self.Imax - self.Imin)
        img.clamp_(0.0, 255.0)
        img = img.byte()  # convert to 8-bit tensor

        return img


class ImageWriter:
    """
    Utility class to write images to disk.
    Also writes the image timestamps into a text file.
    """

    def __init__(self, options):

        self.output_folder = options.output_folder
        self.dataset_name = options.dataset_name
        self.save_events = options.show_events
        self.event_display_mode = options.event_display_mode
        self.num_bins_to_show = options.num_bins_to_show
        if options.verbose:
            print('== Image Writer ==')
        if self.output_folder:
            ensure_dir(self.output_folder)
            ensure_dir(join(self.output_folder, self.dataset_name))
            if options.verbose:
                print('Will write images to: {}'.format(join(self.output_folder, self.dataset_name)))
            # self.timestamps_file = open(join(self.output_folder, self.dataset_name, 'timestamps.txt'), 'a')

            if self.save_events:
                self.event_previews_folder = join(self.output_folder, self.dataset_name, 'events')
                ensure_dir(self.event_previews_folder)
                if options.verbose:
                    print('Will write event previews to: {}'.format(self.event_previews_folder))

            atexit.register(self.__cleanup__)
        else:
            print('Will not write images to disk.')

    def __call__(self, img, event_tensor_id, events=None):
        if not self.output_folder:
            return
        if self.save_events and events is not None:
            event_preview = make_event_preview(events, mode=self.event_display_mode,
                                               num_bins_to_show=self.num_bins_to_show)
            cv2.imwrite(join(self.event_previews_folder,
                             'events_{:019d}.png'.format(event_tensor_id)), event_preview)

        cv2.imwrite(join(self.output_folder, self.dataset_name,
                         '{:019d}.png'.format(event_tensor_id)), img)


    def __cleanup__(self):
        return 0


class UnsharpMaskFilter:
    """
    Utility class to perform unsharp mask filtering on reconstructed images.
    """

    def __init__(self, options, device):
        self.unsharp_mask_amount = options.unsharp_mask_amount
        self.unsharp_mask_sigma = options.unsharp_mask_sigma
        self.gaussian_kernel_size = 5
        self.gaussian_kernel = gkern(self.gaussian_kernel_size,
                                     self.unsharp_mask_sigma).unsqueeze(0).unsqueeze(0).to(device)

    def __call__(self, img):
        if self.unsharp_mask_amount > 0:

            self.gaussian_kernel = self.gaussian_kernel.type_as(img)

            blurred = F.conv2d(img, self.gaussian_kernel,
                                padding=self.gaussian_kernel_size // 2)
            img = (1 + self.unsharp_mask_amount) * img - self.unsharp_mask_amount * blurred
        return img


class ImageFilter:
    """
    Utility class to perform some basic filtering on reconstructed images.
    """

    def __init__(self, options):
        self.bilateral_filter_sigma = options.bilateral_filter_sigma

    def __call__(self, img):

        if self.bilateral_filter_sigma:
            filtered_img = np.zeros_like(img)
            filtered_img = cv2.bilateralFilter(
                img, 5, 25.0 * self.bilateral_filter_sigma, 25.0 * self.bilateral_filter_sigma)
            img = filtered_img

        return img


def optimal_crop_size(max_size, max_subsample_factor):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ReflectionPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = np.expand_dims(X[dy, :], axis=0)
    elif dy < 0:
        X[dy:, :] = np.expand_dims(X[dy, :], axis=0)
    if dx > 0:
        X[:, :dx] = np.expand_dims(X[:, dx], axis=1)
    elif dx < 0:
        X[:, dx:] = np.expand_dims(X[:, dx], axis=1)
    return X


def upsample_color_image(grayscale_highres, color_lowres_bgr, colorspace='LAB'):
    """
    Generate a high res color image from a high res grayscale image, and a low res color image,
    using the trick described in:
    http://www.planetary.org/blogs/emily-lakdawalla/2013/04231204-image-processing-colorizing-images.html
    """
    assert(len(grayscale_highres.shape) == 2)
    assert(len(color_lowres_bgr.shape) == 3 and color_lowres_bgr.shape[2] == 3)

    if colorspace == 'LAB':
        # convert color image to LAB space
        lab = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2LAB)
        # replace lightness channel with the highres image
        lab[:, :, 0] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=lab, code=cv2.COLOR_LAB2BGR)
    elif colorspace == 'HSV':
        # convert color image to HSV space
        hsv = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2HSV)
        # replace value channel with the highres image
        hsv[:, :, 2] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=hsv, code=cv2.COLOR_HSV2BGR)
    elif colorspace == 'HLS':
        # convert color image to HLS space
        hls = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2HLS)
        # replace lightness channel with the highres image
        hls[:, :, 1] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=hls, code=cv2.COLOR_HLS2BGR)

    return color_highres_bgr


def merge_channels_into_color_image(channels):
    """
    Combine a full resolution grayscale reconstruction and four color channels at half resolution
    into a color image at full resolution.

    :param channels: dictionary containing the four color reconstructions (at quarter resolution),
                     and the full resolution grayscale reconstruction.
    :return a color image at full resolution
    """

    assert('R' in channels)
    assert('G' in channels)
    assert('W' in channels)
    assert('B' in channels)
    assert('grayscale' in channels)

    # upsample each channel independently
    for channel in ['R', 'G', 'W', 'B']:
        channels[channel] = cv2.resize(channels[channel], dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Shift the channels so that they all have the same origin
    channels['B'] = shift_image(channels['B'], dx=1, dy=1)
    channels['G'] = shift_image(channels['G'], dx=1, dy=0)
    channels['W'] = shift_image(channels['W'], dx=0, dy=1)

    # reconstruct the color image at half the resolution using the reconstructed channels RGBW
    reconstruction_bgr = np.dstack([channels['B'],
                                    cv2.addWeighted(src1=channels['G'], alpha=0.5,
                                                    src2=channels['W'], beta=0.5,
                                                    gamma=0.0, dtype=cv2.CV_8U),
                                    channels['R']])

    reconstruction_grayscale = channels['grayscale']

    # combine the full res grayscale resolution with the low res to get a full res color image
    upsampled_img = upsample_color_image(reconstruction_grayscale, reconstruction_bgr)
    
    return upsampled_img