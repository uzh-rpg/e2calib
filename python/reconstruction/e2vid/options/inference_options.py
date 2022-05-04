import os

E2VID_PATH = os.path.join(os.path.dirname(__file__), "../pretrained/E2VID_lightweight.pth.tar")

def set_inference_options(parser):
    parser.add_argument('--verbose', '-v',  action='store_true', default=False, help='Verbose output')


    parser.add_argument('-c', '--path_to_model', type=str,
                        help='path to the model weights',
                        default=E2VID_PATH)

    parser.add_argument('-o', '--output_folder', default='frames', type=str)  # if None, will not write the images to disk
    parser.add_argument('--dataset_name', default='e2calib', type=str)

    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--gpu_id',  type=int, default=0)
    
    parser.add_argument('--use_fp16', action='store_true')
    
    """ Display """
    parser.add_argument('--display', action='store_true')

    parser.add_argument('--no-display_trackbars', dest='no_display_trackbars', action='store_true',
                        default=False)
    parser.set_defaults(no_display_trackbars=False)

    parser.add_argument('--no-show_reconstruction', dest='no_show_reconstruction', action='store_true',
                        default=False)
    parser.set_defaults(no_show_reconstruction=False)

    parser.add_argument('--show_events', dest='show_events', action='store_true')

    parser.add_argument('--event_display_mode', default='red-blue', type=str,
                        help="Event display mode ('red-blue' or 'grayscale')")

    parser.add_argument('--num_bins_to_show', default=-1, type=int,
                        help="Number of bins of the voxel grid to show when displaying events (-1 means show all the bins).")

    parser.add_argument('--display_border_crop', default=0, type=int,
                        help="Remove the outer border of size display_border_crop before displaying image.")

    parser.add_argument('--display_wait_time', default=1, type=int,
                        help="Time to wait after each call to cv2.imshow, in milliseconds (default: 1)")

    """ Post-processing / filtering """

    # (optional) path to a text file containing the locations of hot pixels to ignore
    parser.add_argument('--hot_pixels_file', default=None, type=str)

    # (optional) unsharp mask
    parser.add_argument('--unsharp_mask_amount', default=0.3, type=float)
    parser.add_argument('--unsharp_mask_sigma', default=1.0, type=float)

    # (optional) bilateral filter
    parser.add_argument('--bilateral_filter_sigma', default=0.0, type=float)

    # (optional) flip the event tensors vertically
    parser.add_argument('--flip', dest='flip', action='store_true')

    """ Tone mapping (i.e. rescaling of the image intensities)"""
    parser.add_argument('--Imin', default=0.0, type=float,
                        help="Min intensity for intensity rescaling (linear tone mapping).")
    parser.add_argument('--Imax', default=1.0, type=float,
                        help="Max intensity value for intensity rescaling (linear tone mapping).")
    parser.add_argument('--auto_hdr', dest='auto_hdr', action='store_true',
                        help="If True, will compute Imin and Imax automatically.")
    parser.set_defaults(auto_hdr=False)
    parser.add_argument('--auto_hdr_median_filter_size', default=10, type=int,
                        help="Size of the median filter window used to smooth temporally Imin and Imax")
    parser.add_argument('--gamma', default=1.5, type=float,
                        help="Gamma correction value.")
    parser.add_argument('--contrast', default=1.0, type=float,
                        help="Contrast correction value.")
    parser.add_argument('--brightness', default=0.0, type=float,
                        help="Brightness correction value.")
    parser.add_argument('--saturation', default=1.0, type=float,
                        help="Saturation correction value.")

    """ Perform color reconstruction? (only use this flag with the DAVIS346color) """
    parser.add_argument('--color', dest='color', action='store_true')

    """ Advanced parameters """
    # disable normalization of input event tensors (saves a bit of time, but may produce slightly worse results)
    parser.add_argument('--no-normalize', dest='no_normalize', action='store_true')

    # disable recurrent connection (will severely degrade the results; for testing purposes only)
    parser.add_argument('--no-recurrent', dest='no_recurrent', action='store_true')

    return parser