import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 400
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 1.0

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2
#SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])
#SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])
specs = [
    SSDSpec(25, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(13, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(7, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(4, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]

priors = generate_ssd_priors(specs, image_size)