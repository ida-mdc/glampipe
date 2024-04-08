from glob import glob
import os
import logging
import numpy as np
import imageio
import tifffile as tif
from glampipe.config import OUTPUT_PATH
from glampipe.config import OUTPUT_PATH_ORIGINAL, OUTPUT_PATH_PROBABILITY, OUTPUT_PATH_PROBABILITY_PROCESSED, \
    OUTPUT_PATH_BINARY, OUTPUT_PATH_MESH, OUTPUT_PATH_TRAINING_SET


def make_output_sub_dirs():
    os.makedirs(OUTPUT_PATH_ORIGINAL)
    os.makedirs(OUTPUT_PATH_PROBABILITY)
    os.makedirs(OUTPUT_PATH_PROBABILITY_PROCESSED)
    os.makedirs(OUTPUT_PATH_BINARY)
    os.makedirs(OUTPUT_PATH_MESH)
    os.makedirs(OUTPUT_PATH_TRAINING_SET)


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


def get_original_image_paths(dir_path, condition):
    all_image_paths = glob(os.path.join(dir_path, '*', '*.lsm'))
    all_image_paths.extend(glob(os.path.join(dir_path, '*', '*', '*.lsm')))
    all_image_paths.extend(glob(os.path.join(dir_path, '*', '*', '*', '*.lsm')))

    if condition is not None:
        str_include, str_exclude = get_string_rules(condition)

        all_image_paths = [p for p in all_image_paths if
                           any(s in p.lower() for s in str_include) and
                           not any(s in p.lower() for s in str_exclude)]

    logging.info(f'Number of input original images: {len(all_image_paths)}')

    return all_image_paths


def get_probability_processed_image_paths():
    all_image_paths = glob(os.path.join(OUTPUT_PATH_PROBABILITY_PROCESSED, '*.tif'))
    return all_image_paths


def read_image(p, mesh_pixel_size_pre_interpolation=None):
    im = tif.imread(p)
    logging.info(f'Image shape {im.shape}')
    if im.ndim != 3:
        logging.warning('Expected a 3D image. Skipping.')
        return False
    elif (mesh_pixel_size_pre_interpolation is not None) and np.any(im.shape < mesh_pixel_size_pre_interpolation):
        logging.warning(f'Image is smaller than needed mesh print size - {mesh_pixel_size_pre_interpolation}')
        return False
    else:
        return im


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


def save_patch_segmentation_images(i_path, i_patch, patch, probability_map):

    tif.imsave(os.path.join(OUTPUT_PATH_ORIGINAL, f'{i_path}_{i_patch}.tif'), patch)
    tif.imsave(os.path.join(OUTPUT_PATH_PROBABILITY, f'{i_path}_{i_patch}.tif'), probability_map)


def save_post_processed_probability_images(i_patch, i_path, largest_object_mask, probability_map_upsampled, thr):
    tif.imsave(os.path.join(OUTPUT_PATH_PROBABILITY_PROCESSED, f'{i_path}_{i_patch}.tif'), probability_map_upsampled)
    tif.imsave(os.path.join(OUTPUT_PATH_BINARY, f'{i_path}_{i_patch}_{thr}.tif'), largest_object_mask)


def image_properties_to_csv(i_path, p, voxel_size, interpolation_factors,
                            mesh_pixel_size_pre_interpolation, mesh_size_micron_str,
                            patches_start_idxs):
    file_path = os.path.join(OUTPUT_PATH, 'image_properties.csv')

    # Open the file in 'a+' mode to append and read;
    with open(file_path, 'a+') as f:
        f.seek(0)  # Move to the start of the file to check if it's empty
        if f.read(1) == "":  # Check if the file is empty
            header = (
                "idx,p,voxel_size,interpolation_factors,"
                "mesh_pixel_size_pre_interpolation,mesh_size_micron_str,"
                "patches_start_idxs\n"
            )
            f.seek(0)  # Move back to the start of the file to write the header
            f.write(header)

        row = (
            f"{i_path},{p},{voxel_size},{interpolation_factors},"
            f"{mesh_pixel_size_pre_interpolation},{mesh_size_micron_str},"
            f"{patches_start_idxs}\n"
        )
        f.write(row)


def save_mesh(mesh, filename):
    filepath = os.path.join(OUTPUT_PATH_MESH, f'{filename}.stl')
    mesh.export(filepath)


def get_binary_image(filename):
    binary_path = glob(os.path.join(OUTPUT_PATH_BINARY, f'{filename}*.tif'))[0]
    binary = tif.imread(binary_path)
    thr = int(binary_path.split('_')[-1].split('.')[0])
    return binary, thr
