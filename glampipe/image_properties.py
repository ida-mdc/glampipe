import tifffile as tif
import logging
import numpy as np
import re
from skimage.filters import threshold_triangle, threshold_otsu, threshold_li, threshold_yen
from glampipe.config import ARGS


def get_magnification(p):
    return re.findall(r'[124]0[xX]', p)[-1][:2]


def get_voxel_size(p):
    with tif.TiffFile(p) as t:
        x = z = None  # Initialize variables

        # Check and extract from LSM metadata if available
        if hasattr(t, 'lsm_metadata') and t.lsm_metadata is not None:
            x = float("%.10f" % t.lsm_metadata["VoxelSizeX"])
            z = float("%.10f" % t.lsm_metadata["VoxelSizeZ"])
        elif hasattr(t, 'imagej_metadata') and 'Info' in t.imagej_metadata:
            # Extract from ImageJ metadata if LSM metadata is not available
            imagej_metadata = t.imagej_metadata['Info']
            for line in imagej_metadata.split('\n'):
                if 'Scaling|Distance|Value #1' in line:
                    x = float(line.split('=')[1].strip())
                elif 'Scaling|Distance|Value #3' in line:
                    z = float(line.split('=')[1].strip())
        else:
            raise ValueError('Voxel size not found in LSM metadata or ImageJ metadata.')

    logging.info(f'Voxel size: x-{x} z-{z}')
    return np.asarray([z, x, x])


def get_interpolation_factor(voxel_size, default_voxel_size):
    interpolation_factors = np.round((voxel_size / default_voxel_size) * 2) / 2

    logging.info(f'Needed interpolation factors (nearest 0.5) - {interpolation_factors}')
    return interpolation_factors


def get_mesh_size_in_pixels_pre_interpolation(default_mesh_size_in_pixels, interpolation_factors):
    mesh_size_in_pixels_pre_interpolation = (default_mesh_size_in_pixels / interpolation_factors).astype(np.uint16)
    logging.info(f'mesh_size_in_pixels_pre_interpolation {mesh_size_in_pixels_pre_interpolation}')

    return mesh_size_in_pixels_pre_interpolation


def get_patches_start_idxs(im_shape, patch_shape):
    """
    Calculate the start indices for non-overlapping patches of a given shape within an image.

    Args:
    - im_shape (tuple): The shape of the image (depth, height, width).
    - patch_shape (tuple): The desired patch shape (depth, height, width).

    Returns:
    - List of tuples representing the start indices of each patch.
    """

    if im_shape[0] < patch_shape[0]:
        return []

    # Calculate the start index for the first dimension so the patch is centered
    z_start = (im_shape[0] - patch_shape[0]) // 2

    # For the second and third dimensions, calculate start indices for valid, non-overlapping patches
    y_starts = list(range(0, im_shape[1] - patch_shape[1] + 1, patch_shape[1]))
    x_starts = list(range(0, im_shape[2] - patch_shape[2] + 1, patch_shape[2]))

    # Generate tuples of start indices for all valid patches
    patches_start_idxs = [np.asarray([z_start, y, x]) for y in y_starts for x in x_starts]

    logging.info(f'Number of patches (meshes) from image: {len(patches_start_idxs)}')

    return patches_start_idxs


def quarters_of_image(im):
    return [im[:im.shape[0]//2, :im.shape[1]//2, :im.shape[2]//2],
            im[im.shape[0]//2:, :im.shape[1]//2, :im.shape[2]//2],
            im[:im.shape[0]//2, im.shape[1]//2:, :im.shape[2]//2],
            im[:im.shape[0]//2, :im.shape[1]//2, im.shape[2]//2:]]


def is_image_too_empty(im, threshold=0.15):

    quarters = quarters_of_image(im)

    is_too_empty = any(np.count_nonzero(q) / q.size < threshold for q in quarters)

    if is_too_empty:
        logging.warning('Patch is empty. Skipping.')

    return is_too_empty


def is_image_too_full(im, threshold=0.75):

    quarters = quarters_of_image(im)

    is_too_full = any(np.count_nonzero(q) / q.size > threshold for q in quarters)

    if is_too_full:
        logging.warning('Patch is full. Skipping.')

    return is_too_full


def get_mesh_size_micron_str(mesh_size_in_pixels_pre_interpolation, voxel_size):
    m_to_micron = 1000000
    mesh_size = [s * float(v) * m_to_micron for s, v in zip(mesh_size_in_pixels_pre_interpolation, voxel_size)]
    mesh_size_str = f'xy{mesh_size[1]:.3f}z{mesh_size[0]:.3f}'

    logging.info(f'Mesh size in microns: {mesh_size_str}')

    return mesh_size_str


def get_threshold(image, method):
    """
    Calculates the threshold of an image using the specified method.

    Parameters:
    - image: np.ndarray, the input image.
    - method: str, the thresholding method to use ('li', 'otsu', 'triangle', 'yen').

    Returns:
    - threshold: float, the calculated threshold value.
    """
    flat_image = image.flatten()

    if method == 'li':
        threshold = threshold_li(flat_image)
    elif method == "otsu":
        threshold = threshold_otsu(flat_image)
    elif method == "triangle":
        threshold = threshold_triangle(flat_image)
    elif method == 'yen':
        threshold = threshold_yen(flat_image)
    else:
        raise ValueError("Threshold method must be one of: 'li', 'otsu', 'triangle', 'yen'")

    logging.info(f'Found threshold {threshold}')

    return threshold


def get_histogram_and_max_percentage(im):
    histogram, bins = np.histogram(im.flatten(), bins=256, range=[0, 256])
    max_count = histogram.max()
    max_percentage = max_count / histogram.sum() * 100

    histogram = histogram / histogram.sum() * 100

    return histogram, bins, max_percentage


def is_histogram_peak_above_threshold(im):
    _, _, max_percentage = get_histogram_and_max_percentage(im)
    return max_percentage > ARGS.histogram_peak_threshold
