from scipy import ndimage
import numpy as np
import logging
from scipy.ndimage import gaussian_filter


def extract_patch(image, patch_start_idxs, patch_size):
    patch = image[patch_start_idxs[0]:patch_start_idxs[0] + patch_size[0],
            patch_start_idxs[1]:patch_start_idxs[1] + patch_size[1],
            patch_start_idxs[2]:patch_start_idxs[2] + patch_size[2]]
    return patch


def enhance_contrast_3d(im):
    # Compute the histogram and CDF of the entire 3D stack
    hist, bins = np.histogram(im.flatten(), bins=256, range=[0, 255])
    cdf = hist.cumsum()  # Cumulative distribution function
    # Normalize the CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')

    # Initialize the array for enhanced images
    enhanced_im = np.zeros_like(im)

    # Apply histogram equalization based on the normalized CDF to each slice
    for i in range(im.shape[0]):
        sl = im[i, :, :]
        # Normalize the slice to use the full range of 0-255
        slice_normalized = np.interp(sl.flatten(), bins[:-1], cdf_normalized)
        enhanced_im[i, :, :] = slice_normalized.reshape(sl.shape)

    return enhanced_im.astype(np.uint8)


def smooth_image(im, sigma=1):
    im = ndimage.gaussian_filter(im, sigma=sigma)
    logging.info(f'Gaussian filter applied - sigma {sigma}')
    return im


def resize_interpolate_image(im, zxy_factors, additional_factor=2):
    zoom_factors = [s * additional_factor for s in zxy_factors]

    if all(f < 1 for f in zoom_factors):  # down-sampling - anti-aliasing
        sigma = [1.0/f for f in zoom_factors]
        im = gaussian_filter(im, sigma=sigma)
    elif any(f < 1 for f in zoom_factors):
        logging.warning(
            f'Applying down-sampling (at least in one dimension) without anti-aliasing. Zoom factor: {zoom_factors}')

    logging.info(f'Up-sampling factor z,x,y - {zoom_factors}')
    zoomed_image = ndimage.zoom(im, zoom_factors, order=3).astype(np.float64)

    logging.info(f'Up-sampled shape and dtype: {zoomed_image.shape}, {zoomed_image.dtype}')

    return zoomed_image


def create_binary(im, thr):
    thr_image = np.where(im > thr, 255, 0).astype(np.uint8)
    logging.info(f'thresholded image')
    labeled_image, num_features = ndimage.label(thr_image)
    logging.info(f'number of objects: {num_features}')
    sizes = ndimage.sum(thr_image, labeled_image, range(num_features + 1))
    largest_label = np.argmax(sizes[1:]) + 1
    largest_object_mask = (labeled_image == largest_label)

    return largest_object_mask
