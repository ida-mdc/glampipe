from scipy import ndimage
import numpy as np
import logging


def extract_patch(image, patch_start_idxs, patch_size):
    patch = image[patch_start_idxs[0]:patch_start_idxs[0] + patch_size[0],
            patch_start_idxs[1]:patch_start_idxs[1] + patch_size[1],
            patch_start_idxs[2]:patch_start_idxs[2] + patch_size[2]]
    return patch


def smooth_image(im, sigma=1):
    im = ndimage.gaussian_filter(im, sigma=sigma)
    logging.info(f'Gaussian filter applied - sigma {sigma}')
    return im


def interpolate_for_upsample(im, zxy_factors, additional_factor=2):
    zoom_factors = [s * additional_factor for s in zxy_factors]
    if any(f < 0 for f in zoom_factors):
        logging.warning(
            f'Applying down-sampling (at least in one dimension) without anti-aliasing. Zoom factor: {zoom_factors}')

    logging.info(f'Up-sampling factor z,x,y - {zoom_factors}')
    upsampled_image = ndimage.zoom(im, zoom_factors, order=3).astype(np.float64)

    logging.info(f'Up-sampled shape and dtype: {upsampled_image.shape}, {upsampled_image.dtype}')

    return upsampled_image


# def interpolate_image(im, voxel_spacing, extra_factor=1, is_downsampling=False, order=3):
#
#     zoom_factors = [s*extra_factor for s in voxel_spacing]
#
#     # Apply Gaussian filter for antialiasing if down-sampling
#     if is_downsampling:
#         sigma = [1.0/f for f in zoom_factors]
#         im = gaussian_filter(im, sigma=sigma)
#
#     sampled_image = zoom(im, zoom_factors, order=order).astype(np.uint8)
#     return sampled_image


def create_binary(im, thr):
    thr_image = np.where(im > thr, 255, 0).astype(np.uint8)
    logging.info(f'thresholded image')
    labeled_image, num_features = ndimage.label(thr_image)
    logging.info(f'number of objects: {num_features}')
    sizes = ndimage.sum(thr_image, labeled_image, range(num_features + 1))
    largest_label = np.argmax(sizes[1:]) + 1
    largest_object_mask = (labeled_image == largest_label)

    return largest_object_mask
