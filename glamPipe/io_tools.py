from glob import glob
import os
import logging
import numpy as np
import imageio
import tifffile as tif
from glamPipe.mesh_operations import save_mesh
from glamPipe.config import ARGS, TODAY_STR


def make_output_dirs():
    output_dir = os.path.join('data', f'{TODAY_STR}_{ARGS.condition}_{ARGS.threshold_method}')

    os.makedirs(os.path.join(output_dir, 'original'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'probability'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'probability_processed'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'binary'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'meshes'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'training_set'), exist_ok=True)

    return output_dir


def get_existing_output_dir():
    output_dir = os.path.join(ARGS.path_output, ARGS.segmentation_dir_date)
    # check if subdir probability if not empty. if empty throw error
    # check if meshes subdir is empty. if its not empty, raise a warning
    # check if training_set subdir is empty. if its not empty, raise a warning
    return output_dir


def get_string_rules(condition):
    if condition == 'emphysema':
        str_include = ['emph', 'lastase']
        str_exclude = ['projection', 'quickoverview', 'healthy', 'quick10xoverview']
    elif condition == 'healthy':
        str_include = ['healthy', 'pbs', 'wt']
        str_exclude = ['projection', 'quickoverview', 'quick10xoverview', 'bleo', 'mphe', 'tile']
    elif condition == 'fibrotic':
        str_include = ['fibrotic', 'bleo']
        str_exclude = ['projection', 'quickoverview', 'quick10xoverview']  # , 'tile']
    else:
        raise ValueError(f'Condition must be in [emphysema, healthy, fibrotic]. Got {condition}.')

    return str_include, str_exclude


def get_image_paths(dir_path, condition, output_dir):
    all_image_paths = glob(os.path.join(dir_path, '*', '*.lsm'))
    all_image_paths.extend(glob(os.path.join(dir_path, '*', '*', '*.lsm')))
    all_image_paths.extend(glob(os.path.join(dir_path, '*', '*', '*', '*.lsm')))

    if condition is not None:
        str_include, str_exclude = get_string_rules(condition)

        all_image_paths = [p for p in all_image_paths if
                           any(s in p.lower() for s in str_include) and
                           not any(s in p.lower() for s in str_exclude)]

    logging.info(f'Number of images: {len(all_image_paths)}')

    save_paths_to_txt(all_image_paths, output_dir)

    return all_image_paths


def save_paths_to_txt(paths, output_dir):
    with open(os.path.join(output_dir, 'images_paths.txt'), 'w') as f:
        for ip, p in enumerate(paths):
            f.write(f'{ip}, {p}\n')


def read_image(p, mesh_size_in_pixels_pre_interpolation):
    im = tif.imread(p)
    logging.info(f'Image shape {im.shape}')
    if im.ndim != 3:
        logging.warning('Expected a 3D image. Skipping.')
        return False
    elif np.any(im.shape < mesh_size_in_pixels_pre_interpolation):
        logging.warning(f'Image is smaller than needed mesh print size - {mesh_size_in_pixels_pre_interpolation}')
        return False
    else:
        return im


def save_images_and_mesh(output_dir, i_path, i_patch, patch, probability_map, largest_object_mask, mesh,
                         mesh_size_micron_str):

    save_mesh(mesh, output_dir, i_path, i_patch, mesh_size_micron_str)


def read_gif(filename):
    # Read the GIF using imageio
    gif = imageio.mimread(filename)

    # Process frames to handle different shapes
    processed_frames = []
    for frame in gif:
        if frame.ndim == 2:  # Frame is 2D
            processed_frames.append(frame)
        elif len(frame.shape) == 3:  # Frame is 3D
            processed_frames.append(frame[:, :, 0])  # Take only the first channel (e.g., red channel)

    # Convert the list of frames to a numpy array
    im = np.stack(processed_frames, axis=0)
    return im


def save_as_gif(im, filename):
    frames = [im[:, :, i] for i in range(im.shape[2])]

    # Convert 2D frames to 3D (RGB + Alpha)
    converted_frames = []
    for frame in frames:
        if frame.ndim == 2:  # Checking for 2D frame
            rgb_frame = np.stack([frame, frame, frame], axis=-1)  # Convert grayscale to RGB
            alpha_channel = np.ones_like(frame) * 255  # Create an alpha channel
            rgba_frame = np.dstack((rgb_frame, alpha_channel))  # Add alpha channel
            converted_frames.append(rgba_frame)
        else:
            logging.error('frame should be 2D at this stage.')
            # converted_frames.append(frame)  # If already 3D, use as is

    imageio.mimsave(filename, converted_frames, format='GIF')


def save_patch_segmentation_images(output_dir, i_path, i_patch,
                                   patch, probability_map, probability_map_upsampled, largest_object_mask):
    tif.imsave(os.path.join(output_dir, 'original', f'{i_path}_{i_patch}.tif'), patch)
    tif.imsave(os.path.join(output_dir, 'probability', f'{i_path}_{i_patch}.tif'), probability_map)
    tif.imsave(os.path.join(output_dir, 'probability_processed', f'{i_path}_{i_patch}.tif'), probability_map_upsampled)
    tif.imsave(os.path.join(output_dir, 'binary', f'{i_path}_{i_patch}.tif'), largest_object_mask)
