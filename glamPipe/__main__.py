import logging
import os
from glamPipe.config import ARGS, set_logger
from glamPipe import io_tools
from glamPipe import image_properties
from glamPipe import segmentation
from glamPipe import mesh_operations
from glamPipe import image_operations


def main():
    set_logger()

    if ARGS.segmentation_dir_date is None:
        output_dir = io_tools.make_output_dirs(ARGS.path_output)
    else:
        output_dir = os.path.join(ARGS.path_output, ARGS.segmentation_dir_date)

    paths = io_tools.get_image_paths(ARGS.path_originals, ARGS.condition)
    io_tools.save_paths_to_txt(paths, output_dir)

    model_resource, prediction_pipeline = segmentation.load_model(ARGS.path_segmentation_model)

    for i_path, p in enumerate(paths):

        logging.info(f'image {i_path}: {p}')

        voxel_size = image_properties.get_voxel_size(p)
        interpolation_factors = image_properties.get_interpolation_factor(voxel_size, ARGS.default_voxel_size)
        mesh_size_in_pixels_pre_interpolation = image_properties.get_mesh_size_in_pixels_pre_interpolation(
            ARGS.default_mesh_size_in_pixels, interpolation_factors)
        mesh_size_micron_str = image_properties.get_mesh_size_micron_str(mesh_size_in_pixels_pre_interpolation,
                                                                         voxel_size)

        im = io_tools.read_image(p, mesh_size_in_pixels_pre_interpolation)
        if isinstance(im, bool):
            continue

        patches_start_idxs = image_properties.get_patches_start_idxs(im.shape, mesh_size_in_pixels_pre_interpolation)

        for i_patch, patch_start_idxs in enumerate(patches_start_idxs):
            patch = image_operations.extract_patch(im, patch_start_idxs, mesh_size_in_pixels_pre_interpolation)

            probability_map = segmentation.predict(patch, model_resource, prediction_pipeline)

            probability_map_smooth = image_operations.smooth_image(probability_map, ARGS.gaussian_sigma)
            probability_map_upsampled = image_operations.interpolate_for_upsample(probability_map_smooth,
                                                                                  interpolation_factors)

            thr = image_properties.get_threshold(probability_map_upsampled, ARGS.threshold_method)
            largest_object_mask = image_operations.create_binary(probability_map_upsampled, thr)

            vertices, faces = mesh_operations.make_mesh(probability_map_upsampled, thr, largest_object_mask)
            mesh = mesh_operations.post_process_mesh(vertices, faces)

            io_tools.save_images_and_mesh(output_dir,
                                          i_path,
                                          i_patch,
                                          patch,
                                          probability_map,
                                          largest_object_mask,
                                          mesh,
                                          mesh_size_micron_str)


if __name__ == "__main__":
    main()
