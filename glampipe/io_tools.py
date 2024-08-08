from glob import glob
import os
import logging
from datetime import datetime
import numpy as np
import imageio
import tifffile as tif
import pandas as pd
from glampipe.config import PROPERTIES_FILE
from glampipe.config import (OUTPUT_PATH,
                             OUTPUT_PATH_BINARY,
                             OUTPUT_PATH_MESH,
                             OUTPUT_PATH_TRAINING_SET)


def make_output_sub_dir(dir_path):
    os.makedirs(dir_path)


def get_string_rules(condition):
    if condition == 'emphysema':
        str_include = ['emph', 'lastase']
        str_exclude = ['projection', 'quickoverview', 'healthy', 'quick10xoverview']
    elif condition == 'healthy':
        str_include = ['healthy', 'pbs', 'wt']
        str_exclude = ['bleo', 'mphe']
    elif condition == 'fibrotic':
        str_include = ['fibrotic', 'bleo']
        str_exclude = ['projection']  # 'quickoverview', 'quick10xoverview']  # , 'tile']
    else:
        raise ValueError(f'Condition must be in [emphysema, healthy, fibrotic]. Got {condition}.')

    return str_include, str_exclude


def get_original_image_paths(dir_path, condition):
    all_image_paths = glob(os.path.join(dir_path, '*', '*.lsm'))
    all_image_paths.extend(glob(os.path.join(dir_path, '*', '*', '*.lsm')))
    all_image_paths.extend(glob(os.path.join(dir_path, '*', '*', '*', '*.lsm')))

    all_image_paths.extend(glob(os.path.join(dir_path, '*2024*', '*.tif')))
    all_image_paths.extend(glob(os.path.join(dir_path, '*', '*2024*', '*.tif')))
    all_image_paths.extend(glob(os.path.join(dir_path, '*', '*', '*2024*', '*.tif')))

    if condition is not None:
        str_include, str_exclude = get_string_rules(condition)

        all_image_paths = [p for p in all_image_paths if
                           any(s in p.lower() for s in str_include) and
                           not any(s in p.lower() for s in str_exclude)]

    logging.info(f'Number of input original images: {len(all_image_paths)}')

    return all_image_paths


def get_image_paths(sub_dir):
    all_image_paths = glob(os.path.join(sub_dir, '*.tif'))
    return all_image_paths


def read_image(p, mesh_pixel_size_pre_interpolation=None):
    im = tif.imread(p)
    im = np.squeeze(im)
    logging.info(f'Image shape {im.shape}')

    if im.ndim == 4:
        im = im[:, 1]

    if im.ndim != 3:
        logging.warning('Expected a 3D image. Skipping.')
        return False
    elif (mesh_pixel_size_pre_interpolation is not None) and np.any(im.shape < mesh_pixel_size_pre_interpolation):
        logging.warning(f'Image is smaller than needed mesh print size - {mesh_pixel_size_pre_interpolation}')
        return False
    else:
        return im


def get_filename(p, is_extension=True):
    filename = os.path.basename(p)
    if not is_extension:
        filename = filename.split('.')[0]
    return filename


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

    im = (im - im.min()) / (im.max() - im.min())
    im = (im * 255).astype('uint8')

    file_path = os.path.join(OUTPUT_PATH_TRAINING_SET, f'{filename}.gif')
    # frames = [im[:, :, i] for i in range(im.shape[2])]
    frames = [im[i, :, :] for i in range(im.shape[0])]

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

    imageio.mimsave(file_path, converted_frames, format='GIF')


def save_image(output_path, filename, image):
    tif.imwrite(os.path.join(output_path, filename), image)


def save_training_set_image(filename, image):
    image = (image*255).astype(np.uint8)
    tif.imwrite(os.path.join(OUTPUT_PATH_TRAINING_SET, f'{filename}.tif'), image)


def get_array_as_string(array):
    return np.array2string(array, separator=' ')[1:-1]


def image_properties_to_csv(i_path, p, voxel_size, interpolation_factors,
                            mesh_pixel_size_pre_interpolation, mesh_size_micron_str,
                            patches_start_idxs):
    file_path = os.path.join(OUTPUT_PATH, PROPERTIES_FILE)

    patches_start_idxs_str = ' '.join([get_array_as_string(array) for array in patches_start_idxs])

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
            f"{i_path},{p},{get_array_as_string(voxel_size)},{get_array_as_string(interpolation_factors)},"
            f"{get_array_as_string(mesh_pixel_size_pre_interpolation)},{mesh_size_micron_str},"
            f"{patches_start_idxs_str}\n"
        )
        f.write(row)


def get_interpolation_factors_from_csv(i_path):
    df = pd.read_csv(os.path.join(OUTPUT_PATH, PROPERTIES_FILE))
    interpolation_factors = df[df.idx == i_path]['interpolation_factors'].values[0]
    interpolation_factors = np.array(list(map(float, interpolation_factors.split())))
    return interpolation_factors


def save_mesh(mesh, filename):
    filepath = os.path.join(OUTPUT_PATH_MESH, f'{filename}.stl')
    mesh.export(filepath)


def get_binary_image(filename):
    binary_path = glob(os.path.join(OUTPUT_PATH_BINARY, f'{filename}*.tif'))[0]
    binary = tif.imread(binary_path)
    thr = float(binary_path.split('_')[-1][:-4])
    return binary, thr


def replace_string_in_file(file_path, old_string, new_string):
    with open(file_path, 'r') as file:
        content = file.read()

    content = content.replace(old_string, new_string)

    with open(file_path, 'w') as file:
        file.write(content)
