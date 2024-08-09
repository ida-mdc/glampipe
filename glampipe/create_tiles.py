import logging
from glampipe.config import ARGS
from glampipe import io_tools
from glampipe import image_operations
from glampipe import image_properties
from glampipe.config import OUTPUT_PATH_ORIGINAL


def create_tiles_and_save_metadata():
    io_tools.make_output_sub_dir(OUTPUT_PATH_ORIGINAL)
    paths = io_tools.get_original_image_paths(ARGS.path_originals, ARGS.condition)

    for i_path, p in enumerate(paths):

        logging.info(f'image {i_path}: {p}')

        voxel_size = image_properties.get_voxel_size(p)
        interpolation_factors = image_properties.get_interpolation_factor(voxel_size, ARGS.default_voxel_size)
        mesh_pixel_size_pre_interpolation = image_properties.get_mesh_size_in_pixels_pre_interpolation(
            ARGS.default_mesh_size_in_pixels, interpolation_factors)
        mesh_size_micron_str = image_properties.get_mesh_size_micron_str(mesh_pixel_size_pre_interpolation,
                                                                         voxel_size)

        im = io_tools.read_image(p, mesh_pixel_size_pre_interpolation)
        if isinstance(im, bool):
            continue

        tiles_start_idxs = image_properties.get_tiles_start_idxs(im.shape, mesh_pixel_size_pre_interpolation)

        io_tools.image_properties_to_csv(i_path, p, voxel_size, interpolation_factors,
                                         mesh_pixel_size_pre_interpolation, mesh_size_micron_str,
                                         tiles_start_idxs)

        for i_tile, tile_start_idxs in enumerate(tiles_start_idxs):

            tile = image_operations.extract_tile(im, tile_start_idxs, mesh_pixel_size_pre_interpolation)

            if image_properties.is_tile_too_empty(tile):
                logging.info(f'Tile {i_path}_{i_tile} is empty-ish. Skipping.')
                continue

            if ARGS.is_enhance_contrast:
                tile = image_operations.enhance_contrast_3d(tile)

            io_tools.save_image(OUTPUT_PATH_ORIGINAL, f'{i_path}_{i_tile}.tif', tile)
