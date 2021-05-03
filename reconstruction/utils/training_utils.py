'''
This code is provided for internal research and development purposes by Huawei solely,
in accordance with the terms and conditions of the research collaboration agreement of May 7, 2020.
Any further use for commercial purposes is subject to a written agreement.
'''
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import cv2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_evenly_spaced_elements(num_elements, sequence_length):
    return [i * sequence_length // num_elements + sequence_length // (2 * num_elements) for i in range(num_elements)]


def flow2rgb(disp_x, disp_y, max_magnitude=None):
    """
    Convert an optic flow tensor to an RGB color map for visualization
    Code adapted from: https://github.com/ClementPinard/FlowNetPytorch/blob/master/main.py#L339

    :param disp_x: a [H x W] NumPy array containing the X displacement
    :param disp_x: a [H x W] NumPy array containing the Y displacement
    :returns bgr: a [H x W x 3] NumPy array containing a color-coded representation of the flow
    """
    assert(disp_x.shape == disp_y.shape)
    H, W = disp_x.shape

    X, Y = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))

    flow_x = (X - disp_x) * float(W) / 2
    flow_y = (Y - disp_y) * float(H) / 2
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)

    if max_magnitude is None:
        v = np.zeros(magnitude.shape, dtype=np.uint8)
        cv2.normalize(src=magnitude, dst=v, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        v = 255.0 * magnitude / max_magnitude
        v = v.astype(np.uint8)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = 0.5 * angle * 180 / np.pi
    hsv[..., 2] = v
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def plot_grad_flow(named_parameters):
    '''
    RETURNING EARLY
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    return
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            # remove the first part (module name) and last part ("weight") of the module name
            n = '.'.join(n.split('.')[1:-1])
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig('/tmp/gradient_flow.pdf')
    plt.close()
