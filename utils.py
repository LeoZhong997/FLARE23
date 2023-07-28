import numpy as np
from skimage import measure


def small_volume_filter_2d(mask, threshold=0):
    mask_size = mask.shape
    new_mask = mask * 1

    if len(mask_size) == 3:
        for i in range(mask_size[0]):
            if np.sum(mask[i]) > 0:
                new_mask[i] = small_volume_filter_2d(mask[i], threshold)

    if len(mask_size) == 2:
        if np.sum(mask) > 0:
            label = measure.label(mask, connectivity=1)
            regions = measure.regionprops(label)
            for region in regions:
                if region.area < threshold:
                    new_mask[label == region.label] = 0

    return new_mask
